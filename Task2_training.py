import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import argparse
import nibabel as nib
import torchvision.transforms as transforms
from monai.networks.nets import ResNet
from monai.networks.layers import Act
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings("ignore")
# ============ device & seed ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class focal_loss(torch.nn.Module):
    def __init__(self, alpha=0.1, gamma=2.0, reduction='mean'):
        """
        alpha: weighting factor for the positive class (float scalar between 0 and 1)
        gamma: focusing parameter to reduce loss for easy examples
        reduction: 'mean', 'sum' or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: raw outputs (no sigmoid)
        # targets: binary labels (0 or 1)

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt = probability of true class

        # Compute alpha factor for each sample
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# ============ clinical preprocess ============
def preprocess_clinical_data(dataframe: pd.DataFrame):
    """
    预处理临床特征。
    这个版本更加稳健，并明确处理数据类型以防止 'numpy.object_' TypeError。
    """
    CONTINUOUS_FEATURES = ["Age", "MTV", "NTV", "T_SUV", "N_SUV", "TLG", "NLG"]
    CATEGORICAL_FEATURES = [
        "Gender", "Tobacco Consumption", "Alcohol Consumption",
        "Performance Status", "M-stage", "Treatment"
    ]

    # --- 1. 单独处理连续特征 ---
    continuous_df = dataframe[CONTINUOUS_FEATURES].copy()
    for col in CONTINUOUS_FEATURES:
        if continuous_df[col].isnull().any():
            median_val = continuous_df[col].median()
            continuous_df[col] = continuous_df[col].fillna(median_val)

    continuous_scaler = StandardScaler()
    # 创建一个包含标准化值的新 DataFrame
    continuous_scaled_df = pd.DataFrame(
        continuous_scaler.fit_transform(continuous_df),
        columns=CONTINUOUS_FEATURES,
        index=dataframe.index  # 保持原始索引，以便后续合并
    )

    # --- 2. 单独处理分类特征 ---
    categorical_df = dataframe[CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        if categorical_df[col].isnull().any():
            categorical_df[col] = categorical_df[col].fillna('Unknown')

    categorical_df = categorical_df.astype(str)

    # 从分类部分创建独热编码 DataFrame
    categorical_encoded_df = pd.get_dummies(
        categorical_df,
        prefix=CATEGORICAL_FEATURES,
        dtype=np.float32  # 使用 float32 以保证兼容性
    )

    # --- 3. 合并处理好的 DataFrame ---
    final_df = pd.concat([continuous_scaled_df, categorical_encoded_df], axis=1)

    # --- 4. 准备最终输出 ---
    # 显式地将最终结果转换为 float32 类型的 NumPy 数组，以保证正确的数据类型
    final_feature_matrix = final_df.values.astype(np.float32)

    processed_features = {}
    for i, patient_id in enumerate(dataframe["PatientID"]):
        processed_features[patient_id] = final_feature_matrix[i]

    preprocessors = {
        'continuous_scaler': continuous_scaler,
        'feature_names': list(final_df.columns),
        'n_features': final_feature_matrix.shape[1]
    }

    return {'features': processed_features, 'preprocessors': preprocessors}

# ============ dataset ============
def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data

def get_mip(data, axis):
    return np.max(data, axis=axis)

class ImageSurvivalDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe_split: pd.DataFrame, config: dict,
                 processed_clinical_features: dict,
                 mode='preprocessed', use_augmentation=False):
        self.dataframe = dataframe_split.reset_index(drop=True)
        self.config = config
        self.mode = mode
        self.use_augmentation = use_augmentation
        self.clinical_features = processed_clinical_features
        if self.mode not in ['preprocessed', 'on_the_fly']:
            raise ValueError("Mode must be either 'preprocessed' or 'on_the_fly'")
        self.transform = None

    def __len__(self):
        return len(self.dataframe)

    def _get_item_from_preprocessed(self, patient_id):
        file_path = os.path.join(self.config['mip_dir'], f"{patient_id}.npy")
        image = torch.from_numpy(np.load(file_path)).float()
        return image

    def _get_item_on_the_fly(self, patient_id):
        # 不走该分支；保留占位，避免大改
        return None

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        pid = row["PatientID"]

        if self.mode == 'preprocessed':
            image = self._get_item_from_preprocessed(pid)
        else:
            image = self._get_item_on_the_fly(pid)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        label = torch.tensor([row["HPV Status"]], dtype=torch.float32)
        clinical = torch.tensor(self.clinical_features[pid], dtype=torch.float32)
        return image, label, clinical

# ============ model ============
class MaskBranch(nn.Module):
    def __init__(self, in_channels=2, out_channels=64):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.branch(x)

class DualBranchFusionResNet(nn.Module):
    def __init__(self, num_classes=1, clinical_input_size=128, dropout_p=0.1):
        super().__init__()
        self.image_backbone = ResNet(
            block="basic", layers=[2,2,2,2], block_inplanes=[32,64,128,256],
            spatial_dims=3, n_input_channels=2, num_classes=num_classes,
            act=Act.PRELU, norm="batch"
        )
        self.img_c1_out = self.image_backbone.conv1.out_channels
        self.mask_branch = MaskBranch(in_channels=2, out_channels=self.img_c1_out)
        self.image_feat_dim = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity()
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_input_size, 64),##from 128 to 64
            nn.ReLU(),
            #nn.BatchNorm1d(64), ##from 128 to 32
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.clin_feat_dim = 32#64#128
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.image_feat_dim + self.clin_feat_dim),
            nn.Linear(self.image_feat_dim + self.clin_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, clinical):
        image_tensor = x[:, :2, ...]
        mask_tensor_int = x[:, 2:3, ...].long()
        mask_one_hot = torch.nn.functional.one_hot(
            mask_tensor_int.squeeze(1), num_classes=3
        ).permute(0,4,1,2,3).float()
        lesion_mask = mask_one_hot[:, 1:, ...]
        mask_features = self.mask_branch(lesion_mask)

        img = self.image_backbone.conv1(image_tensor)
        img = self.image_backbone.bn1(img)
        img = self.image_backbone.act(img)

        if mask_features.shape[2:] != img.shape[2:]:
            mask_features = F.interpolate(mask_features, size=img.shape[2:], mode="trilinear", align_corners=False)
        fused = img + mask_features

        x = self.image_backbone.maxpool(fused)
        x = self.image_backbone.layer1(x)
        x = self.image_backbone.layer2(x)
        x = self.image_backbone.layer3(x)
        x = self.image_backbone.layer4(x)
        x = self.image_backbone.avgpool(x)
        x = torch.flatten(x, 1)
        #print('img feature size',x.shape)

        c = self.clinical_processor(clinical)
        feat = torch.cat([x, c], dim=1)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        return logits  # logits（不加 sigmoid）

# ============ train / eval（不搜阈值） ============
def train_one_epoch(loader, model, criterion, optimizer, device, bce_raw=None, w_neg=None, w_pos=None):
    model.train()
    running_loss = 0.
    all_labels, all_outputs = [], []

    for images, labels, clinical in loader:
        if isinstance(images, np.ndarray): images = torch.from_numpy(images)
        if isinstance(clinical, np.ndarray): clinical = torch.from_numpy(clinical)
        images = images.float().to(device, non_blocking=True)
        clinical = clinical.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images, clinical)

        if bce_raw is not None and w_neg is not None and w_pos is not None:
            loss = criterion(logits, labels)
            # 逐样本加权 BCE：让 0 类更重
            #loss_elems = bce_raw(logits, labels)              # [B,1] or [B]
            #weights = torch.where(labels > 0.5, w_pos, w_neg) # y=1->w_pos, y=0->w_neg
            #loss = (loss_elems * weights).mean()
        else:
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        all_outputs.extend(probs)
        all_labels.extend(labels.detach().cpu().numpy().ravel())

    train_loss = running_loss / len(loader.dataset)
    train_auc = roc_auc_score(all_labels, all_outputs)
    preds_05 = (np.array(all_outputs) >= 0.5).astype(int)  # 训练期用 0.5 做参考日志
    train_bacc = balanced_accuracy_score(all_labels, preds_05)
    return train_loss, train_auc, train_bacc

@torch.no_grad()
def evaluate_epoch_05(loader, model, criterion):
    model.eval()
    running_loss = 0.
    all_labels, all_outputs = [], []

    for images, labels, clinical in loader:
        if isinstance(images, np.ndarray): images = torch.from_numpy(images)
        if isinstance(clinical, np.ndarray): clinical = torch.from_numpy(clinical)
        images = images.float().to(device, non_blocking=True)
        clinical = clinical.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, clinical)
        loss = criterion(logits, labels)
        running_loss += loss.item() * labels.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        all_outputs.extend(probs)
        all_labels.extend(labels.detach().cpu().numpy().ravel())

    val_loss = running_loss / len(loader.dataset)
    val_auc = roc_auc_score(all_labels, all_outputs)
    preds_05 = (np.array(all_outputs) >= 0.5).astype(int)  # 训练期仅记录 t=0.5 的指标（不搜阈值）
    val_bacc = balanced_accuracy_score(all_labels, preds_05)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds_05, labels=[0,1]).ravel()
    val_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    val_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity ← 新增
    #print([i.item() for i in all_outputs], [i.item() for i in all_labels], np.sum([i.item() for i in all_labels]))
    return val_loss, val_auc, val_bacc, val_spec, val_sens  # ← 多返回一个

# ============ Temperature Scaling ============
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))  # T = exp(0) = 1

    def forward(self, logits):
        T = torch.exp(self.log_temp)
        return logits / T

def fit_temperature(model, loader, device, max_iter=200, lr=0.01):
    model.eval()
    scaler = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)
    bce = nn.BCEWithLogitsLoss()

    logits_list, labels_list = [], []
    with torch.no_grad():
        for images, labels, clinical in loader:
            if isinstance(images, np.ndarray): images = torch.from_numpy(images)
            if isinstance(clinical, np.ndarray): clinical = torch.from_numpy(clinical)
            images = images.float().to(device)
            clinical = clinical.float().to(device)
            labels = labels.to(device)
            logits = model(images, clinical)
            logits_list.append(logits.detach())
            labels_list.append(labels.detach())
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    def closure():
        opt.zero_grad()
        scaled = scaler(logits)
        loss = bce(scaled, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return scaler

@torch.no_grad()
def collect_val_logits_labels(model, loader):
    model.eval()
    logits_list, labels_list = [], []
    for images, labels, clinical in loader:
        if isinstance(images, np.ndarray): images = torch.from_numpy(images)
        if isinstance(clinical, np.ndarray): clinical = torch.from_numpy(clinical)
        images = images.float().to(device)
        clinical = clinical.float().to(device)
        labels = labels.to(device)
        logits = model(images, clinical)
        logits_list.append(logits.detach())
        labels_list.append(labels.detach())
    return torch.cat(logits_list, 0), torch.cat(labels_list, 0)

def pick_threshold_youden(logits: torch.Tensor, labels: torch.Tensor, scaler: TemperatureScaler = None):
    if scaler is not None:
        logits = scaler(logits)
    probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    y_true = labels.cpu().numpy().ravel().astype(int)
    fpr, tpr, ts = roc_curve(y_true, probs)
    youden = tpr - fpr
    t_best = ts[np.argmax(youden)]
    return float(t_best)

# ============ main ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_img", type=float, default=None)
    parser.add_argument("--lr_clin", type=float, default=None)
    parser.add_argument("--fold", type=int, nargs='*', default=None)
    args, _ = parser.parse_known_args()

    DATA_LOADING_MODE = 'preprocessed'
    config = {
        'csv_path': '/projects/pet_ct_challenge/survival_code/HECKTOR_2025_Training_with_Radiomics_Features_task3.csv',
        'mip_dir': '/processing/l.cai/3D_crop',
        'img_base_paths': {},
        'label_base_paths': {},
        'output_dir': '/projects/pet_ct_challenge/l.cai/task3_experiments/clinical_model_5fold_output_task3',
        'target_size': (310, 310),
        'n_splits': 5,
        'epochs': 30,
        'batch_size': 8,
        'lr': 5e-5,
        'weight_decay': 1e-2,
        'use_augmentation': False
    }

    SEED = 3407
    set_seed(SEED)
    os.makedirs(config['output_dir'], exist_ok=True)
    print(f"Using device: {device}")
    print(f"Data loading mode: {DATA_LOADING_MODE}")

    try:
        full_df = pd.read_csv(config['csv_path'])
    except FileNotFoundError:
        print(f"CSV not found: {config['csv_path']}")
        return

    proc = preprocess_clinical_data(full_df)
    features = proc['features']
    input_size = proc['preprocessors']['n_features']
    print(f"Clinical feature dim: {input_size}")

    kfold = KFold(n_splits=config['n_splits'], shuffle=True, random_state=SEED)
    all_splits = list(kfold.split(full_df))

    folds_to_run = args.fold if args.fold is not None else range(config['n_splits'])
    fold_results_auc, fold_results_bacc, fold_results_spec = [], [], []

    for fold in folds_to_run:
        tr_idx, va_idx = all_splits[fold]
        print("-"*50); print(f"--- FOLD {fold+1}/{config['n_splits']} ---"); print("-"*50)

        train_df = full_df.iloc[tr_idx]
        val_df = full_df.iloc[va_idx]

        train_ds = ImageSurvivalDataset(train_df, config, features, mode='preprocessed')
        val_ds   = ImageSurvivalDataset(val_df,   config, features, mode='preprocessed')

        # WeightedRandomSampler（均衡采样）
        y_train = train_df["HPV Status"].values.astype(int)
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y_train]


        num_workers = 16

        num_samples = (len(sample_weights) // config['batch_size']) * config['batch_size']
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,  # ← 不用 len(sample_weights)
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,  # ← 关键：丢弃最后不足一个 batch 的样本
        )

        val_loader = DataLoader(val_ds, batch_size=config['batch_size'],
                                shuffle=False, num_workers=num_workers, pin_memory=True)

        model = DualBranchFusionResNet(num_classes=1, clinical_input_size=input_size).to(device)
        lr_img = args.lr_img if args.lr_img is not None else config['lr']
        lr_clin = args.lr_clin if args.lr_clin is not None else config['lr']
        img_params  = list(model.image_backbone.parameters()) + list(model.mask_branch.parameters())
        clin_params = list(model.clinical_processor.parameters()) + list(model.classifier.parameters())
        optimizer = torch.optim.AdamW(
            [{"params": img_params,  "lr": lr_img},
             {"params": clin_params, "lr": lr_clin}],
            weight_decay=config['weight_decay']
        )

        # 验证日志用标准 BCE
        n0 = int(class_counts[0]) if len(class_counts) > 0 else 0
        n1 = int(class_counts[1]) if len(class_counts) > 1 else 0
        pos_weight = torch.tensor([n0 / n1], device=device)
        print('pos_weight',pos_weight)
        criterion = focal_loss() #nn.BCEWithLogitsLoss(pos_weight=pos_weight)#focal_loss()#nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 逐样本加权 BCE（让 0 类更重）
        n0 = int(class_counts[0]) if len(class_counts) > 0 else 0
        n1 = int(class_counts[1]) if len(class_counts) > 1 else 0
        w_neg = torch.tensor(float(n1 / max(n0, 1)), device=device, dtype=torch.float32)  # 0类权重
        w_pos = torch.tensor(1.0, device=device, dtype=torch.float32)                      # 1类权重
        bce_raw = nn.BCEWithLogitsLoss(reduction='none')
        print(f"[Fold {fold+1}] class_counts={class_counts.tolist()}, w_neg={w_neg.item():.3f}")

        best_val_bacc = -1.0
        best_val_auc  = 0.0
        best_val_spec = 0.0
        best_epoch    = -1

        for epoch in range(1, config['epochs'] + 1):
            t0 = time.time()
            tr_loss, tr_auc, tr_bacc = train_one_epoch(
                train_loader, model, criterion, optimizer, device,
                bce_raw=bce_raw, w_neg=w_neg, w_pos=w_pos
            )
            va_loss, va_auc, va_bacc, va_spec, val_sens = evaluate_epoch_05(val_loader, model, criterion)
            t1 = time.time()

            print(f"Epoch {epoch}/{config['epochs']} | {t1-t0:.2f}s | "
                  f"Train Loss {tr_loss:.4f} AUC {tr_auc:.4f} BalAcc {tr_bacc:.4f} | "
                  f"Val Loss {va_loss:.4f} AUC {va_auc:.4f} BalAcc {va_bacc:.4f} Spec {va_spec:.4f}  Sens {val_sens:.4f}(t=0.5)")

            # 训练期不搜阈值，用 BalAcc(0.5) 做保存标准（也可换成 AUC）
            if va_bacc > best_val_bacc:
                best_val_bacc = va_bacc
                best_val_auc  = va_auc
                best_val_spec = va_spec
                best_epoch    = epoch
                torch.save({'model': model.state_dict()},
                           os.path.join(config['output_dir'], f'best_model_weights_fold_{fold+1}.pth'))
                print(f"  -> New best Val BalAcc(0.5): {best_val_bacc:.4f} (epoch {best_epoch}) [weights saved]")

        # ===== 每折训练结束后：温度标定 + Youden’s J 选阈值 =====
        # 载入刚保存的最佳权重（确保用最佳状态做标定/阈值选择）
        best_path = os.path.join(config['output_dir'], f'best_model_weights_fold_{fold+1}.pth')
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])

        # 1) 收集验证集 logits/labels
        val_logits, val_labels = collect_val_logits_labels(model, val_loader)

        # 2) 温度标定（可直接用整个 val_loader；若有单独 calibration split 更佳）
        scaler = fit_temperature(model, val_loader, device)
        T_value = torch.exp(scaler.log_temp).item()

        # 3) Youden’s J 选阈值（在标定后的概率上）
        t_best = pick_threshold_youden(val_logits, val_labels, scaler)

        # 4) 一起保存：权重 + 温度 + 阈值 + 指标
        final_ckpt_path = os.path.join(config['output_dir'], f'best_model_fold_{fold+1}.pth')
        torch.save({
            'model': model.state_dict(),
            'temperature': T_value,
            'best_threshold': float(t_best),
            'val_bal_acc_at_05': float(best_val_bacc),
            'val_auc': float(best_val_auc),
            'val_specificity_at_05': float(best_val_spec),
            'best_epoch': int(best_epoch),
            'fold': int(fold+1),
        }, final_ckpt_path)
        print(f"[Fold {fold+1}] Saved final ckpt with temperature={T_value:.4f}, threshold={t_best:.3f} -> {final_ckpt_path}")

        fold_results_auc.append(best_val_auc)
        fold_results_bacc.append(best_val_bacc)
        fold_results_spec.append(best_val_spec)

    # ===== summary =====
    print("="*50)
    print("5-Fold Summary (metrics @ t=0.5 during training)")
    print("="*50)
    for i in range(len(fold_results_auc)):
        print(f"Fold {i+1}: AUC={fold_results_auc[i]:.4f}, "
              f"BalAcc(0.5)={fold_results_bacc[i]:.4f}, Spec(0.5)={fold_results_spec[i]:.4f}")
    print("\n--- Averages ---")
    print(f"AUC:  {np.mean(fold_results_auc):.4f} ± {np.std(fold_results_auc):.4f}")
    print(f"BalAcc(0.5): {np.mean(fold_results_bacc):.4f} ± {np.std(fold_results_bacc):.4f}")
    print(f"Spec(0.5):   {np.mean(fold_results_spec):.4f} ± {np.std(fold_results_spec):.4f}")

if __name__ == '__main__':
    main()







































# import os
# import random
# import time
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from sklearn.model_selection import KFold
# from torch.utils.data import DataLoader
# import argparse
# # ==================== 新增的导入 ====================
# import nibabel as nib
# import torchvision.models as models
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import timm # 需要安装timm库: pip install timm
# import torch
# import torch.nn as nn
# # MONAI是用于医学影像AI的优秀PyTorch库，推荐使用
# # 如果没有安装，请 pip install monai
# from monai.networks.nets import ResNet
# from monai.networks.layers import Act
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, balanced_accuracy_score
# from sklearn.metrics import confusion_matrix
# # ===================================================
#
# # ==============================================================================
# # 0. 设置和辅助函数 (与之前完全相同)
# # ==============================================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#
#
# # nll_loss, concordance_index_, cal_ci, _calculate_risk, prediction_nll
# # 这些函数保持原样，无需改动。为保持代码简洁，此处省略。
# # 你可以将它们直接复制过来。
#
# def nll_loss(h, y, c, alpha=0.5, eps=1e-7, reduction='sum'):
#     y = y.type(torch.int64)
#     c = c.type(torch.int64)
#     hazards = torch.sigmoid(h)
#     S = torch.cumprod(1 - hazards, dim=1)
#     S_padded = torch.cat([torch.ones_like(c), S], 1)
#     s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
#     h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
#     s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=eps)
#     uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
#     censored_loss = - c * torch.log(s_this)
#     neg_l = censored_loss + uncensored_loss
#     if alpha is not None:
#         loss = (1 - alpha) * neg_l + alpha * uncensored_loss
#     if reduction == 'mean':
#         loss = loss.mean()
#     else:
#         loss = loss.sum()
#     return loss
#
#
# def concordance_index_(y_true, y_pred):
#     time_value = torch.squeeze(y_true[0:, 0])
#     event = torch.squeeze(y_true[0:, 1]).type(torch.bool)
#     time_1 = time_value.unsqueeze(1).expand(1, time_value.size()[0], time_value.size()[0]).squeeze()
#     event_1 = event.unsqueeze(1).expand(1, event.size()[0], event.size()[0]).squeeze()
#     ix = torch.where(torch.logical_and(time_1 < time_value, event_1))
#     s1 = y_pred[ix[0]]
#     s2 = y_pred[ix[1]]
#     ci = torch.mean((s1 < s2).float())
#     return ci
#
#
# def cal_ci(y, pred):
#     ci_dict = {}
#     for target_class in range(int(y.shape[1] / 2)):
#         ci_dict[target_class] = concordance_index_(y[:, target_class * 2:(target_class + 1) * 2].to(device),
#                                                    -pred.to(device))
#     return ci_dict
#
#
# def _calculate_risk(h):
#     hazards = torch.sigmoid(h)
#     survival = torch.cumprod(1 - hazards, dim=1)
#     risk = -torch.sum(survival, dim=1)
#     return risk, survival
#
#
# def prediction_nll(model, dl):
#     model.eval()
#     with torch.no_grad():
#         j = 0
#         for i, data_batch in enumerate(dl):
#             # 修改点：现在 data_batch 是 (images, labels)
#             images, labels, clinical_info = data_batch
#             images = images.to(device)
#             labels = labels.to(device)
#             clinical_info=clinical_info.to(device)
#             pred = model(images,clinical_info)
#             y_batch = labels
#             if j == 0:
#                 pred_all = pred
#                 y_all = y_batch
#                 j = 1
#             else:
#                 pred_all = torch.cat([pred_all, pred])
#                 y_all = torch.cat([y_all, y_batch])
#         risk, _ = _calculate_risk(pred_all)
#         # y_all[:, 1] = 1 - y_all[:, 1]
#         ci_dict = cal_ci(y_all, risk)
#     return ci_dict, pred_all
#
#
# # ==============================================================================
# # 1. 新的 Dataset 类
# # ==============================================================================
#
# def load_nifti(path):
#     """加载Nifti文件并返回数据数组"""
#     img = nib.load(path)
#     data = img.get_fdata()
#     return data
#
#
# def get_mip(data, axis):
#     """计算指定轴上的最大强度投影"""
#     return np.max(data, axis=axis)
#
# def preprocess_clinical_data(dataframe: pd.DataFrame):
#     """
#     预处理临床特征。
#     这个版本更加稳健，并明确处理数据类型以防止 'numpy.object_' TypeError。
#     """
#     CONTINUOUS_FEATURES = ["Age", "MTV", "NTV", "T_SUV", "N_SUV", "TLG", "NLG"]
#     CATEGORICAL_FEATURES = [
#         "Gender", "Tobacco Consumption", "Alcohol Consumption",
#         "Performance Status", "M-stage", "Treatment"
#     ]
#
#     # --- 1. 单独处理连续特征 ---
#     continuous_df = dataframe[CONTINUOUS_FEATURES].copy()
#     for col in CONTINUOUS_FEATURES:
#         if continuous_df[col].isnull().any():
#             median_val = continuous_df[col].median()
#             continuous_df[col] = continuous_df[col].fillna(median_val)
#
#     continuous_scaler = StandardScaler()
#     # 创建一个包含标准化值的新 DataFrame
#     continuous_scaled_df = pd.DataFrame(
#         continuous_scaler.fit_transform(continuous_df),
#         columns=CONTINUOUS_FEATURES,
#         index=dataframe.index  # 保持原始索引，以便后续合并
#     )
#
#     # --- 2. 单独处理分类特征 ---
#     categorical_df = dataframe[CATEGORICAL_FEATURES].copy()
#     for col in CATEGORICAL_FEATURES:
#         if categorical_df[col].isnull().any():
#             categorical_df[col] = categorical_df[col].fillna('Unknown')
#
#     categorical_df = categorical_df.astype(str)
#
#     # 从分类部分创建独热编码 DataFrame
#     categorical_encoded_df = pd.get_dummies(
#         categorical_df,
#         prefix=CATEGORICAL_FEATURES,
#         dtype=np.float32  # 使用 float32 以保证兼容性
#     )
#
#     # --- 3. 合并处理好的 DataFrame ---
#     final_df = pd.concat([continuous_scaled_df, categorical_encoded_df], axis=1)
#
#     # --- 4. 准备最终输出 ---
#     # 显式地将最终结果转换为 float32 类型的 NumPy 数组，以保证正确的数据类型
#     final_feature_matrix = final_df.values.astype(np.float32)
#
#     processed_features = {}
#     for i, patient_id in enumerate(dataframe["PatientID"]):
#         processed_features[patient_id] = final_feature_matrix[i]
#
#     preprocessors = {
#         'continuous_scaler': continuous_scaler,
#         'feature_names': list(final_df.columns),
#         'n_features': final_feature_matrix.shape[1]
#     }
#
#     return {'features': processed_features, 'preprocessors': preprocessors}
#
# class ImageSurvivalDataset(torch.utils.data.Dataset):
#     def __init__(self, dataframe_split: pd.DataFrame, config: dict,
#                  processed_clinical_features: dict,
#                  mode='preprocessed', use_augmentation=False):
#         self.dataframe = dataframe_split.reset_index(drop=True)
#         self.config = config
#         self.mode = mode
#         self.use_augmentation = use_augmentation
#         self.clinical_features = processed_clinical_features  # 新增 — 临床特征字典
#         # 验证模式是否有效
#         if self.mode not in ['preprocessed', 'on_the_fly']:
#             raise ValueError("Mode must be either 'preprocessed' or 'on_the_fly'")
#
#         if self.use_augmentation:
#             self.transform = transforms.Compose([
#                 transforms.ToPILImage(),  # 增强操作通常在PIL Image上进行
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomRotation(10),
#                 transforms.ToTensor()  # 转回Tensor
#             ])
#         else:
#             self.transform = None
#
#     def __len__(self):
#         return len(self.dataframe)
#
#     def _get_item_from_preprocessed(self, patient_id):
#         """从 .npy 文件加载数据"""
#         file_path = os.path.join(self.config['mip_dir'], f"{patient_id}.npy")
#         # if not os.path.exists(file_path):
#         #     print(file_path)
#         #     print(f"Warning: Preprocessed file not found for {patient_id}. Skipping.")
#         #     return None
#         image_4c = torch.from_numpy(np.load(file_path))
#         return image_4c
#
#     def _get_item_on_the_fly(self, patient_id):
#         """实时处理数据"""
#         # ... 查找、加载、MIP、填充、堆叠 ...
#         # 返回 image_4c (Tensor)
#         # --- 查找文件 ---
#         img_base_paths = self.config['img_base_paths']
#         label_base_paths = self.config['label_base_paths']
#         target_size = self.config['target_size']
#
#         img_path_t1 = os.path.join(img_base_paths['task1'], f"{patient_id}_0001.nii.gz")
#         img_path = img_path_t1 if os.path.exists(img_path_t1) else os.path.join(img_base_paths['task2'],
#                                                                                 f"{patient_id}_0001.nii.gz")
#
#         label_path_t1 = os.path.join(label_base_paths['task1'], f"{patient_id}.nii.gz")
#         label_path = label_path_t1 if os.path.exists(label_path_t1) else os.path.join(label_base_paths['task2'],
#                                                                                       f"{patient_id}.nii.gz")
#
#         if not os.path.exists(img_path) or not os.path.exists(label_path):
#             return None
#
#         # --- 处理 ---
#         pet_data = load_nifti(img_path)
#         mask_data = load_nifti(mask_path)
#         if pet_data is None or mask_data is None: return None
#
#         pet_mip_ax0 = get_mip(pet_data, axis=0)
#         pet_mip_ax1 = get_mip(pet_data, axis=1)
#         mask_mip_ax0 = get_mip(mask_data, axis=0)
#         mask_mip_ax1 = get_mip(mask_data, axis=1)
#
#         h, w = pet_mip_ax0.shape
#         pad_h, pad_w = target_size[0] - h, target_size[1] - w
#         pad_top, pad_left = pad_h // 2, pad_w // 2
#         pad_bottom, pad_right = pad_h - pad_top, pad_w - pad_left
#         padding = [pad_left, pad_top, pad_right, pad_bottom]
#
#         image_10c = torch.stack([
#             F.pad(torch.from_numpy(pet_mip_ax0), padding),
#             F.pad(torch.from_numpy(mask_mip_ax0), padding),
#             F.pad(torch.from_numpy(pet_mip_ax1), padding),
#             F.pad(torch.from_numpy(mask_mip_ax1), padding)
#         ], dim=0).float()
#
#         return image_4c
#
#     def __getitem__(self, idx: int):
#         patient_row = self.dataframe.iloc[idx]
#         patient_id = patient_row["PatientID"]
#
#         if self.mode == 'preprocessed':
#             image_10c = self._get_item_from_preprocessed(patient_id)
#         else:  # 'on_the_fly'
#             image_10c = self._get_item_on_the_fly(patient_id)
#
#         if image_10c is None:
#             # 如果找不到文件，递归调用下一个索引，以跳过坏数据
#             return self.__getitem__((idx + 1) % len(self))
#         hpv_status = patient_row["HPV Status"]
#
#         # 标签应该是 float tensor，以便BCEWithLogitsLoss计算
#         label = torch.tensor([hpv_status], dtype=torch.float32)
#         clinical_info_np = self.clinical_features[patient_id]
#         clinical_info = torch.tensor(clinical_info_np, dtype=torch.float32)
#         return image_10c, label,clinical_info
#
#
# # ==============================================================================
# # 2. 新的模型定义
# # ==============================================================================
# class MaskBranch(nn.Module):
#     """
#     一个极简的3D CNN，专门用于处理Mask输入。
#     它只包含一个卷积层，用于将Mask转换为与主干网络匹配的特征图。
#     """
#     def __init__(self, in_channels=2, out_channels=64):
#         super(MaskBranch, self).__init__()
#         # 目标：将 (B, 2, 96, 96, 96) 的Mask处理成 (B, 64, 96, 96, 96) 的特征图
#         # 我们使用一个卷积层来完成通道扩展和特征学习
#         # kernel_size=3, stride=1, padding=1 的组合可以保持空间尺寸不变
#         self.branch = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.branch(x)
# class DualBranchFusionResNet(nn.Module):
#     """
#     图像(CT+PET+Mask) + 临床特征联合模型
#     - 图像：CT/PET 进入 MONAI ResNet；Mask -> MaskBranch；在 conv1+bn+act 后相加融合
#     - 临床：简单 MLP 处理；与图像全局池化后的向量拼接；分类头输出离散生存时间bins的logits
#     """
#     def __init__(self, num_classes=4, clinical_input_size=128, dropout_p=0.1):
#         super().__init__()
#
#         # 1) 图像主干（MONAI 1.5.0 用 n_input_channels）
#         self.image_backbone = ResNet(
#             block="basic", layers=[2, 2, 2, 2], block_inplanes=[32, 64, 128, 256],
#             spatial_dims=3, n_input_channels=2, num_classes=num_classes,
#             act=Act.PRELU, norm="batch"
#         )
#         # conv1 输出通道数，用于对齐 mask 分支
#         self.img_c1_out = self.image_backbone.conv1.out_channels
#
#         # 2) Mask 分支：输入 one-hot 的肿瘤/淋巴结两通道
#         self.mask_branch = MaskBranch(in_channels=2, out_channels=self.img_c1_out)
#
#         # 3) 去掉原来的 fc，保留 image feature 向量
#         self.image_feat_dim = self.image_backbone.fc.in_features
#         self.image_backbone.fc = nn.Identity()
#
#         # 4) 临床特征处理分支（可替代你原 SimpleNN 的前两层）
#         self.clinical_processor = nn.Sequential(
#             nn.Linear(clinical_input_size, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             # nn.Linear(64, 32),
#             # nn.ReLU(),
#             # nn.BatchNorm1d(32),
#         )
#         self.clin_feat_dim = 128
#
#         # 5) 分类头：拼接(图像特征 + 临床特征)后输出 num_classes
#         self.dropout = nn.Dropout(p=dropout_p)
#         self.classifier = nn.Sequential(
#             nn.BatchNorm1d(self.image_feat_dim + self.clin_feat_dim),
#             nn.Linear(self.image_feat_dim + self.clin_feat_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(p=dropout_p),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, x, clinical):
#         """
#         x: [B, 3, D, H, W]  (前2通道=CT+PET，第3通道=mask整数标签(0/1/2))
#         clinical: [B, clinical_input_size]
#         """
#         # --- 图像分支 ---
#         image_tensor = x[:, :2, ...]          # [B,2,D,H,W]
#         mask_tensor_int = x[:, 2:3, ...].long()  # [B,1,D,H,W]
#
#         # mask -> one-hot 后取(肿瘤/淋巴结)二通道
#         mask_one_hot = torch.nn.functional.one_hot(mask_tensor_int.squeeze(1), num_classes=3).permute(0, 4, 1, 2,
#                                                                                                       3).float()
#         lesion_mask = mask_one_hot[:, 1:, ...]                               # [B,2,D,H,W]
#
#         # mask 特征
#         mask_features = self.mask_branch(lesion_mask)                        # [B,C1,d?,h?,w?]
#
#         # 图像 conv1+bn+act
#         img = self.image_backbone.conv1(image_tensor)
#         img = self.image_backbone.bn1(img)
#         img = self.image_backbone.act(img)                                   # [B,C1,D',H',W']
#
#         # 对齐空间尺寸再融合
#         if mask_features.shape[2:] != img.shape[2:]:
#             mask_features = F.interpolate(mask_features, size=img.shape[2:], mode="trilinear", align_corners=False)
#
#         fused = img + mask_features
#
#         # 后续残块
#         x = self.image_backbone.maxpool(fused)
#         x = self.image_backbone.layer1(x)
#         x = self.image_backbone.layer2(x)
#         x = self.image_backbone.layer3(x)
#         x = self.image_backbone.layer4(x)
#         x = self.image_backbone.avgpool(x)
#         x = torch.flatten(x, 1)  # [B, image_feat_dim]
#
#         # --- 临床分支 ---
#         c = self.clinical_processor(clinical)  # [B, clin_feat_dim]
#
#         # --- 融合 + 分类 ---
#         feat = torch.cat([x, c], dim=1)       # [B, image_feat_dim+clin_feat_dim]
#         feat = self.dropout(feat)
#         logits = self.classifier(feat)        # [B, num_classes] （离散时间 bins 的 logits）
#
#         return logits
#
#
#
# # ==============================================================================
# # 3. 训练和评估函数 (适配后的版本)
# # ==============================================================================
#
# def train_one_epoch(loader, model, criterion, optimizer, device):
#     model.train()  # ← 只保留这句
#     running_loss = 0.
#     all_labels, all_outputs = [], []
#
#     for data_batch in loader:
#         images, labels, clinical_info = data_batch
#
#         # 统一 dtype + device
#         if isinstance(images, np.ndarray):
#             images = torch.from_numpy(images)
#         if isinstance(clinical_info, np.ndarray):
#             clinical_info = torch.from_numpy(clinical_info)
#
#         images = images.float().to(device, non_blocking=True)
#         clinical_info = clinical_info.float().to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)
#
#         optimizer.zero_grad()
#         outputs = model(images, clinical_info)
#         loss = criterion(outputs, labels)
#
#         loss.backward()
#         optimizer.step()
#
#         batch_size = labels.size(0)
#         running_loss += loss.item() * batch_size
#
#         all_labels.extend(labels.detach().cpu().numpy().ravel())
#         all_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy().ravel())
#
#     train_loss = running_loss / len(loader.dataset)
#     train_auc = roc_auc_score(all_labels, all_outputs)
#
#     preds_binary = (np.array(all_outputs) >= 0.5).astype(int)
#     train_b_acc = balanced_accuracy_score(all_labels, preds_binary)
#
#     return train_loss, train_auc, train_b_acc
#
#
# def evaluate_model(loader, model, criterion):
#     model.eval()
#     running_loss = 0.
#     all_labels, all_probs = [], []
#
#     with torch.no_grad():
#         for images, labels, clinical_info in loader:
#             if isinstance(images, np.ndarray):
#                 images = torch.from_numpy(images)
#             if isinstance(clinical_info, np.ndarray):
#                 clinical_info = torch.from_numpy(clinical_info)
#
#             images = images.float().to(device, non_blocking=True)
#             clinical_info = clinical_info.float().to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)
#
#             outputs = model(images, clinical_info)          # logits
#             loss = criterion(outputs, labels)
#             running_loss += loss.item() * labels.size(0)
#
#             probs = torch.sigmoid(outputs).detach().cpu().numpy().ravel()
#             all_probs.append(probs)
#             all_labels.append(labels.detach().cpu().numpy().ravel())
#
#     all_probs = np.concatenate(all_probs)
#     all_labels = np.concatenate(all_labels)
#     val_loss = running_loss / len(loader.dataset)
#     val_auc = roc_auc_score(all_labels, all_probs)
#
#     # —— 阈值搜索：最大化 Bal Acc ——
#     ts = np.linspace(0.0, 1.0, 501)
#     best_t, best_bal = 0.5, -1.0
#     for t in ts:
#         preds = (all_probs >= t).astype(int)
#         bal = balanced_accuracy_score(all_labels, preds)
#         if bal > best_bal:
#             best_bal, best_t = bal, t
#
#     # 用最佳阈值计算 specificity（特异度）
#     final_preds = (all_probs >= best_t).astype(int)
#     tn, fp, fn, tp = confusion_matrix(all_labels, final_preds, labels=[0, 1]).ravel()
#     val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#     print("True labels:", all_labels)
#     print("Predicted labels:", final_preds)
#     return val_loss, val_auc, best_bal, val_specificity, best_t
#
#
# # ==============================================================================
# # 4. 主训练流程 (已修改)
# # ==============================================================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--lr_img", type=float, default=None, help="learning rate for image branch")
#     parser.add_argument("--lr_clin", type=float, default=None, help="learning rate for clinical branch")
#     parser.add_argument("--fold", type=int, nargs='*', default=None,
#                         help="Specify which fold(s) to run, e.g., --fold 0 2 4. If not provided, all folds are run.")
#     args, _ = parser.parse_known_args()
#     # --- 主要开关 ---
#     # 'preprocessed': 读取预处理好的 .npy 文件 (速度快)
#     # 'on_the_fly': 实时处理 .nii.gz 文件 (速度慢)
#     DATA_LOADING_MODE = 'preprocessed'
#
#     # --- 统一配置字典 ---
#     config = {
#         'csv_path': '/projects/pet_ct_challenge/survival_code/HECKTOR_2025_Training_with_Radiomics_Features_task3.csv',
#         'mip_dir': '/processing/x.liang/3D_crop',  # 预处理文件目录
#         'img_base_paths': {
#             'task1': '/projects/pet_ct_challenge/l.cai/preprocess/Dataset104_Task1_crop/imagesTr',
#             'task2': '/projects/pet_ct_challenge/l.cai/preprocess/Task2/imagesTr'
#         },
#         'label_base_paths': {
#             'task1': '/projects/pet_ct_challenge/l.cai/preprocess/Dataset104_Task1_crop/labelsTr',
#             'task2': '/projects/pet_ct_challenge/task2/label_All'
#         },
#         'output_dir': '/projects/pet_ct_challenge/survival_code/clinical_model_5fold_output_task3',
#         'target_size': (310, 310),
#         'n_splits': 5,
#         'epochs': 30,
#         'batch_size': 16,
#         'lr': 5*1e-5,#5*1e-5 241日志
#         'weight_decay': 1e-2,
#         'use_augmentation': False
#     }
#
#     # --- 基础设置 ---
#     SEED = 3407
#     set_seed(SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     print(f"Data loading mode: {DATA_LOADING_MODE}")
#     os.makedirs(config['output_dir'], exist_ok=True)
#
#     # --- 1. 加载和预处理数据 ---
#     try:
#         # 修正点: 使用 config['csv_path']
#         full_dataframe = pd.read_csv(config['csv_path'])
#     except FileNotFoundError:
#         print(f"Error: CSV file not found at {config['csv_path']}. Exiting.")
#         return
#
#     print("CSV data loaded successfully.")
#     # clinical 相关的代码
#     processed_data = preprocess_clinical_data(full_dataframe)
#     processed_features = processed_data['features']
#     preprocessors = processed_data['preprocessors']
#     input_size = preprocessors['n_features']
#     print(f"Data preprocessed. Number of features: {input_size}")
#     # --- 2. 5折交叉验证设置 ---
#     # 修正点: 使用 config['n_splits']
#     kfold = KFold(n_splits=config['n_splits'], shuffle=True, random_state=SEED)
#     # 将 kfold.split 的结果转换为列表，以便通过索引访问
#     all_splits = list(kfold.split(full_dataframe))
#
#     # ==================== 关键修改点 2: 根据 fold 参数决定循环范围 ====================
#     # 如果用户没有提供 --fold 参数 (args.fold is None)，则运行所有折
#     # 如果用户提供了，则只运行指定的折
#     folds_to_run = args.fold if args.fold is not None else range(config['n_splits'])
#     fold_results_auc, fold_results_b_acc, fold_results_spec = [], [], []
#     for fold in folds_to_run:
#         train_ids, val_ids = all_splits[fold]  # 从预先生成的列表中获取该折的数据
#         # ===============================================================================
#
#         # fold 是从0开始的索引，打印时我们用 fold + 1
#         print("-" * 50)
#         print(f"--- FOLD {fold + 1}/{config['n_splits']} ---")
#         print("-" * 50)
#
#         # --- 3. 准备当前折的数据 ---
#         train_df = full_dataframe.iloc[train_ids]
#         val_df = full_dataframe.iloc[val_ids]
#
#         # 使用新的 ImageSurvivalDataset，传入完整的config
#         train_dataset = ImageSurvivalDataset(train_df, config, processed_features, mode='preprocessed')
#         val_dataset = ImageSurvivalDataset(val_df, config, processed_features, mode='preprocessed')
#
#         # 修正点: 使用 config 中的参数
#         train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16,
#                                   pin_memory=True)
#         val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16,
#                                 pin_memory=True)
#
#         # --- 4. 初始化模型、优化器和损失函数 ---
#         model = DualBranchFusionResNet(num_classes=1,clinical_input_size=input_size).to(device)
#         # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
#         lr_img = args.lr_img if args.lr_img is not None else config['lr']
#         lr_clin = args.lr_clin if args.lr_clin is not None else config['lr']
#         img_params = list(model.image_backbone.parameters()) + list(model.mask_branch.parameters())
#         clin_params = list(model.clinical_processor.parameters()) + list(model.classifier.parameters())
#
#         optimizer = torch.optim.AdamW(
#             [
#                 {"params": img_params, "lr": lr_img},
#                 {"params": clin_params, "lr": lr_clin},
#             ],
#             weight_decay=config['weight_decay'],
#         )
#         criterion = nn.BCEWithLogitsLoss()
#
#         best_val_bal = -1.0
#         best_val_auc = 0.0
#         best_spec_at_best_bal = 0.0
#         best_threshold = 0.5
#
#         for epoch in range(1, config['epochs'] + 1):
#             start_time = time.time()
#             train_loss, train_auc, train_b_acc = train_one_epoch(train_loader, model, criterion, optimizer, device)
#             val_loss, val_auc, val_b_acc, val_specificity, val_best_t = evaluate_model(val_loader, model, criterion)
#             end_time = time.time()
#
#             print(f"Epoch {epoch}/{config['epochs']} | Time: {end_time - start_time:.2f}s | "
#                   f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Bal Acc: {train_b_acc:.4f} | "
#                   f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Bal Acc: {val_b_acc:.4f}, "
#                   f"Spec: {val_specificity:.4f}, Thr*: {val_best_t:.3f}")
#
#             # —— 以 Bal Acc 作为主保存标准 ——
#             if val_b_acc > best_val_bal:
#                 best_val_bal = val_b_acc
#                 best_val_auc = val_auc
#                 best_spec_at_best_bal = val_specificity
#                 best_threshold = val_best_t
#
#                 torch.save({
#                     'model': model.state_dict(),
#                     'best_threshold': best_threshold,
#                     'epoch': epoch,
#                     'val_bal_acc': best_val_bal,
#                     'val_auc': best_val_auc,
#                     'val_specificity': best_spec_at_best_bal,
#                 }, os.path.join(config['output_dir'], f'best_model_fold_{fold + 1}.pth'))
#                 print(f"  -> New best Val Bal Acc: {best_val_bal:.4f} @ thr={best_threshold:.3f}. Model saved.")
#
#             # 保存每个fold的最佳结果
#         fold_results_auc.append(best_val_auc)
#         fold_results_b_acc.append(best_b_acc_at_best_auc)
#         fold_results_spec.append(best_spec_at_best_auc)
#
#         print(
#             f"\nBest Val Metrics for Fold {fold + 1}: AUC={best_val_auc:.4f}, Bal Acc={best_b_acc_at_best_auc:.4f}, Spec={best_spec_at_best_auc:.4f}\n")
#
#         # --- 总结并报告结果 ---
#     print("=" * 50)
#     print("5-Fold Cross-Validation Summary (at best Val Balanced Accuracy)")
#     print("=" * 50)
#     for i in range(config['n_splits']):
#         print(f"Fold {i + 1}: AUC={fold_results_auc[i]:.4f}, "
#               f"Bal Acc={fold_results_b_acc[i]:.4f}, Spec={fold_results_spec[i]:.4f}")
#
#     # 计算并打印平均值
#     mean_auc = np.mean(fold_results_auc)
#     std_auc = np.std(fold_results_auc)
#     mean_b_acc = np.mean(fold_results_b_acc)
#     std_b_acc = np.std(fold_results_b_acc)
#     mean_spec = np.mean(fold_results_spec)
#     std_spec = np.std(fold_results_spec)
#
#     print("\n--- Average Performance Metrics ---")
#     print(f"Average Val AUC: {mean_auc:.4f} ± {std_auc:.4f}")
#     print(f"Average Val Balanced Accuracy: {mean_b_acc:.4f} ± {std_b_acc:.4f}")
#     print(f"Average Val Specificity: {mean_spec:.4f} ± {std_spec:.4f}")
#
#
# if __name__ == '__main__':
#     main()