import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import SimpleITK as sitk
from scipy.ndimage import label as ndimage_label
import pickle
import json
import nibabel as nib
import torchvision.models as models
from torchvision.transforms import functional as F
import glob
from monai.networks.nets import ResNet
from monai.networks.layers import Act
# ==============================================================================
# 0. 从训练脚本中复用必要的定义
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- 复用模型定义 ---
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
            nn.Linear(clinical_input_size, 64),
            nn.ReLU(),
            #nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.clin_feat_dim = 32
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

        c = self.clinical_processor(clinical)
        feat = torch.cat([x, c], dim=1)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        return logits  # logits（不加 sigmoid）

# 定义特征
CONTINUOUS_FEATURES = ["Age", "MTV", "NTV", "T_SUV", "N_SUV", "TLG", "NLG"]
CATEGORICAL_FEATURES = [
    "Gender", "Tobacco Consumption", "Alcohol Consumption",
    "Performance Status", "M-stage", "Treatment"
]



def build_and_save_preprocessors(
    csv_path='/projects/pet_ct_challenge/survival_code/HECKTOR_2025_Training_with_Radiomics_Features_task3.csv',
    output_json_path='/projects/pet_ct_challenge/survival_code/json/preprocessor_config_task3.json'
):
    """
    修正版：确保Scaler的拟合行为与旧函数完全一致。
    先用中位数填充缺失值，然后再拟合Scaler。
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = df.columns.str.replace('\xa0', ' ').str.strip()

    # --- 连续特征 ---
    df_continuous = df[CONTINUOUS_FEATURES]

    # 1. 计算中位数: 用于填充训练和推理时的缺失值
    median_values = df_continuous.median().to_dict()

    # 2. 填充缺失值: 创建一个被填充后的DataFrame用于拟合
    df_continuous_filled = df_continuous.fillna(median_values)

    # 3. 拟合 Scaler: 在填充后的数据上进行拟合
    scaler = StandardScaler()
    scaler.fit(df_continuous_filled) # <--- 关键修正点

    scaler_params = {
        "mean": dict(zip(CONTINUOUS_FEATURES, scaler.mean_)),
        "scale": dict(zip(CONTINUOUS_FEATURES, scaler.scale_))
    }

    # --- 类别特征 (这部分逻辑不变) ---
    cat_df_train = df[CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        missing_value_strings = ["", "NaN", "nan", "null", "N/A", "NA"]
        is_missing = cat_df_train[col].isna() | cat_df_train[col].isin(missing_value_strings)
        cat_df_train.loc[is_missing, col] = "Unknown"
    cat_df_train = cat_df_train.astype(str)
    dummies = pd.get_dummies(cat_df_train, prefix=CATEGORICAL_FEATURES, dtype=np.float32)
    all_cat_columns = dummies.columns.tolist()

    # --- 保存配置 ---
    config = {
        "scaler": scaler_params,
        "median": median_values, # 中位数仍然需要保存，用于推理
        "onehot_columns": all_cat_columns,
        "continuous_features": CONTINUOUS_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES
    }
    output_dir = os.path.dirname(output_json_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"[✅ DONE] 修正后的预处理器配置已成功生成并保存至:\n{output_json_path}")


#build_and_save_preprocessors()


def preprocess_clinical_data_for_inference_docker(json_path, pet_path, mask_path,
                                                 preprocessor_config_path='/projects/pet_ct_challenge/survival_code/json/preprocessor_config_task3.json'):
    """
    最终确认版：严格按照生成的配置文件进行推理时的数据预处理。
    """
    # 1. 加载配置和 JSON 输入
    with open(json_path, 'r', encoding='utf-8') as f:
        patient_data = json.load(f)
    with open(preprocessor_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    median = config['median']
    CONTINUOUS_FEATURES = config['continuous_features']
    scaler_mean = config['scaler']['mean']
    scaler_scale = config['scaler']['scale']

    # 2. 获取所有连续特征的原始值 (未填充、未标准化)
    #    - 先用一个明确的“缺失”标记（如 np.nan）初始化
    #    - 然后尝试从JSON和图像中填充
    raw_cont_values = {key: np.nan for key in CONTINUOUS_FEATURES}

    if "Age" in patient_data and patient_data["Age"] is not None:
        raw_cont_values["Age"] = float(patient_data["Age"])

    try:
        mask_itk = sitk.ReadImage(mask_path)
        pet_itk = sitk.ReadImage(pet_path)
        mask_array = sitk.GetArrayFromImage(mask_itk)
        pet_array = sitk.GetArrayFromImage(pet_itk)
        spacing = mask_itk.GetSpacing()
        voxel_volume_cm3 = (spacing[0] * spacing[1] * spacing[2]) / 1000.0

        # 提取并赋值，只在掩码存在时进行
        if np.any(mask_array == 1):
            tumor_mask = (mask_array == 1)
            raw_cont_values["MTV"] = np.sum(tumor_mask) * voxel_volume_cm3
            raw_cont_values["T_SUV"] = np.max(pet_array[tumor_mask])
            raw_cont_values["TLG"] = raw_cont_values["MTV"] * np.mean(pet_array[tumor_mask])

        if np.any(mask_array == 2):
            node_mask = (mask_array == 2)
            _, num_lesions = ndimage_label(node_mask)
            raw_cont_values["NTV"] = np.sum(node_mask) * voxel_volume_cm3
            # raw_cont_values["N"] = min(num_lesions, 9)
            raw_cont_values["N_SUV"] = np.max(pet_array[node_mask])
            raw_cont_values["NLG"] = raw_cont_values["NTV"] * np.mean(pet_array[node_mask])

    except Exception as e:
        print(f"警告: 图像特征提取失败 ({e})。相关特征将使用中位数填充。")

    # 3. 【关键】填充缺失值：遍历所有获取到的原始值，如果是NaN，则用配置文件中的中位数替换
    filled_cont_values = {}
    for feature in CONTINUOUS_FEATURES:
        if np.isnan(raw_cont_values[feature]):
            filled_cont_values[feature] = median[feature]
        else:
            filled_cont_values[feature] = raw_cont_values[feature]

    # 4. 标准化：使用填充后的值和从配置文件加载的 scaler 参数
    scaled_cont_values = {}
    for feature in CONTINUOUS_FEATURES:
        scaled_cont_values[feature] = (filled_cont_values[feature] - scaler_mean[feature]) / scaler_scale[feature]

    df_cont = pd.DataFrame([scaled_cont_values], dtype=np.float32)

    # ... 类别特征部分逻辑不变，因为它已经对齐了 ...
    CATEGORICAL_FEATURES = config['categorical_features']
    onehot_columns = config['onehot_columns']
    all_columns = CONTINUOUS_FEATURES + onehot_columns

    cat_values = {}
    for col in CATEGORICAL_FEATURES:
        val = patient_data.get(col, "Unknown")
        if pd.isna(val) or val in ["", "NaN", "nan", "null", "N/A", "NA"]:
            val = "Unknown"
        cat_values[col] = str(val)
    df_cat = pd.DataFrame([cat_values])
    df_cat_dummies = pd.get_dummies(df_cat, prefix=CATEGORICAL_FEATURES, dtype=np.float32)

    df_combined = pd.concat([df_cont, df_cat_dummies], axis=1)
    df_aligned = df_combined.reindex(columns=all_columns, fill_value=0)

    return df_aligned.values.astype(np.float32), len(all_columns)

def load_all_models(model_dir, n_splits, input_size):
    """
    加载所有交叉验证折数训练好的模型。
    """
    models = []
    print("Loading models...")
    for fold in range(1, n_splits + 1):
        model_path = os.path.join(model_dir, f'best_model_fold_{fold}.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found for fold {fold} at {model_path}")

        model = DualBranchFusionResNet(num_classes=1,clinical_input_size=input_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 设置为评估模式
        models.append(model)
        print(f"  -> Loaded model for fold {fold}")
    return models


def predict_risk_ensemble(preprocessed_data, models, device='cuda'):
    """
    使用模型集成对新数据进行推理，并返回最终预测。
    此版本采用“硬投票”（Majority Voting）的策略。
    """
    # 确保输入是正确的形状，例如 (1, num_features)
    data_tensor = torch.tensor(preprocessed_data, dtype=torch.float32).to(device)

    # 用于收集每个模型的 0/1 预测
    all_predictions_binary = []
    # 用于调试，收集每个模型的概率
    all_probabilities_debug = []

    threshold = 0.5

    with torch.no_grad():
        for i, model in enumerate(models):
            # 1. 每个模型进行预测，得到原始输出 (logits)
            logits = model(data_tensor)

            # 2. 将 logits 转换为概率
            probabilities = torch.sigmoid(logits)

            # 3. 将概率转换为 0/1 的二元预测 (硬判断)
            #    (probabilities >= threshold) 会得到一个布尔张量 (True/False)
            #    .int() 将其转换为 0/1
            binary_prediction = (probabilities >= threshold).int()

            # 收集每个模型的硬判断结果
            # .cpu().numpy() 出来的形状会是 (num_samples, 1)
            all_predictions_binary.append(binary_prediction.cpu().numpy())

            # (可选) 收集概率用于调试或计算平均概率
            all_probabilities_debug.append(probabilities.cpu().numpy())

            print(
                f"  -> Model {i + 1} Prediction: {'Positive (1)' if binary_prediction.item() == 1 else 'Negative (0)'} "
                f"(Prob: {probabilities.item():.4f})")

    # 4. 进行投票
    # all_predictions_binary 是一个 list of arrays, 形状类似 [[1], [0], [1], [1], [0]]
    predictions_array = np.array(all_predictions_binary).flatten()

    # 使用 scipy.stats.mode 找到票数最多的那个结果 (0 或 1)
    # mode() 返回众数和它的出现次数
    final_prediction_value, count = stats.mode(predictions_array, keepdims=False)

    # 如果票数相等 (例如 2票True, 2票False)，stats.mode 会返回较小的那个值，即 0 (False)
    # 5. 计算最终的平均概率（作为参考信息）
    mean_probability = np.mean(np.array(all_probabilities_debug))

    # 6. 将最终的投票结果 (0或1) 转换为布尔值 (False/True)
    final_prediction_bool = bool(final_prediction_value)

    print("\n--- Voting Results ---")
    print(f"Individual Votes: {predictions_array}")
    print(f"Final Vote Result (most frequent): {final_prediction_value} ({count} votes)")

    return final_prediction_bool, mean_probability
# ==============================================================================
# 2. 主推理函数
# ==============================================================================
def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata()
class InferenceModel:
    """
    Ensemble inference class that loads multiple fold models and combines predictions.
    Expects checkpoints saved like:
    torch.save({
        'model': model.state_dict(),
        'temperature': T_value,          # optional, default=1.0
        'best_threshold': float(t_best), # optional, default=0.5
        ...
    }, path)
    """

    def __init__(self, model_paths, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_paths = model_paths
        self.models_info = self._load_models()  # list of dicts: {'model': model, 'T': float, 'thr': float}
        print(f"Loaded {len(self.models_info)} models for ensemble inference.")

    def _load_models(self):
        loaded = []
        for path in self.model_paths:
            # 读取 ckpt
            ckpt = torch.load(path, map_location=self.device)

            # 实例化模型结构
            model = DualBranchFusionResNet(num_classes=1, clinical_input_size=input_size).to(self.device)

            # 兼容两种保存方式：字典里有 'model'，或直接是 state_dict
            state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()

            # 取温度与阈值（没有则给默认）
            T = float(ckpt.get('temperature', 1.0)) if isinstance(ckpt, dict) else 1.0
            thr = float(ckpt.get('best_threshold', 0.5)) if isinstance(ckpt, dict) else 0.5

            loaded.append({'model': model, 'T': T, 'thr': thr})
        return loaded

    @torch.no_grad()
    def predict(self, images, clinical):
        """
        images: numpy.ndarray 或 torch.Tensor
        clinical: numpy.ndarray 或 torch.Tensor
        Returns:
            dict(
                prob=float,          # 所有模型温度缩放后概率的平均
                pred=bool,           # 投票后的最终预测
                votes=list[bool],    # 每个模型的单独预测
                thresholds=list,     # 每个模型使用的阈值
                probs=list[float]    # 每个模型的单独概率
            )
        """
        # --- 图像张量 ---
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        images = images.float()
        if images.dim() == 4:  # [C,D,H,W] → [1,C,D,H,W]
            images = images.unsqueeze(0)
        images = images.to(self.device, non_blocking=True)

        # --- 临床张量 ---
        if isinstance(clinical, np.ndarray):
            clinical = torch.from_numpy(clinical)
        clinical = clinical.float()
        if clinical.dim() == 1:  # [F] → [1,F]
            clinical = clinical.unsqueeze(0)
        clinical = clinical.to(self.device, non_blocking=True)

        # --- 投票集成 ---
        probs = []
        votes = []
        thresholds = []

        for mi in self.models_info:
            m, T, thr = mi['model'], mi['T'], mi['thr']
            logit = m(images, clinical)  # [1,1]
            logit_T = logit / T  # 温度缩放
            prob = torch.sigmoid(logit_T).item()  # 转为概率
            pred = prob >= thr  # bool

            probs.append(float(prob))
            votes.append(bool(pred))
            thresholds.append(thr)

        # 最终投票（多数表决）
        vote_sum = sum(votes)
        final_pred = vote_sum > len(votes) / 2  # 多数为 True 才判 True

        # 也返回所有模型概率的平均值作为参考
        avg_prob = float(np.mean(probs))

        return {
            'prob': avg_prob,
            'pred': bool(final_pred),
            'votes': votes,
            'thresholds': thresholds,
            'probs': probs
        }


def preprocess_ct(ct_patch: np.ndarray, min_val: int = -1000, max_val: int = 3000) -> np.ndarray:
    """对CT patch进行Clip和[0,1]归一化。"""
    ct_patch = np.clip(ct_patch, min_val, max_val)
    ct_patch = (ct_patch - min_val) / (max_val - min_val)
    return ct_patch
def find_lesion_centroid(mask_data: np.ndarray) -> np.ndarray:
    """
    计算mask中标签1和2的组合质心。
    如果不存在，则返回图像的几何中心。
    """
    # 找到所有标签为1或2的体素的坐标 (D, H, W)
    coords = np.argwhere((mask_data == 1) | (mask_data == 2))

    if coords.shape[0] > 0:
        # 如果找到了病灶，计算坐标的平均值（即质心）
        centroid = np.mean(coords, axis=0)
        return np.round(centroid).astype(int)
    else:
        # 如果没有病灶，返回图像的几何中心
        shape = mask_data.shape
        center_d = shape[0] // 2
        center_h = shape[1] // 2
        center_w = shape[2] // 2
        return np.array([center_d, center_h, center_w])


def crop_around_center(image_3d: np.ndarray, center_coords: np.ndarray, patch_size: tuple) -> np.ndarray:
    """
    以给定的中心点裁剪3D图像，通过预先填充来处理边界情况。
    """
    patch_d, patch_h, patch_w = patch_size

    # 为了处理边界，我们创建一个足够大的、用0填充的画布
    # 填充量为patch size的一半，可以保证中心点在任何位置都能完整裁剪
    pad_d, pad_h, pad_w = patch_d // 2, patch_h // 2, patch_w // 2

    padded_image = np.pad(
        image_3d,
        ((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )

    # 将原始中心点坐标映射到填充后的大图像上
    padded_center_d = center_coords[0] + pad_d
    padded_center_h = center_coords[1] + pad_h
    padded_center_w = center_coords[2] + pad_w

    # 计算在填充后图像上的裁剪起始坐标
    start_d = padded_center_d - (patch_d // 2)
    start_h = padded_center_h - (patch_h // 2)
    start_w = padded_center_w - (patch_w // 2)

    # 计算结束坐标
    end_d = start_d + patch_d
    end_h = start_h + patch_h
    end_w = start_w + patch_w

    # 从填充后的大图像中裁剪出最终的patch
    cropped_patch = padded_image[start_d:end_d, start_h:end_h, start_w:end_w]

    return cropped_patch
if __name__ == '__main__':
    name = "CHUS-036"

    json_path = f"/projects/pet_ct_challenge/survival_code/json/input_task3/{name}.json"#'/projects/pet_ct_challenge/survival_code/json/input_task3/CHUM-053.json'
    # 构造输出路径（将 input 替换为 output）
    json_output_path =f"/home/l.cai/STU-Net/task3_test/output/rfs.json"#json_path.replace('/input_task3/', '/output_task3/')
    pet_path = f'/projects/pet_ct_challenge/l.cai/preprocess/Dataset104_Task1_crop/imagesTr/{name}_0001.nii.gz'
    ct_path = f'/projects/pet_ct_challenge/l.cai/preprocess/Dataset104_Task1_crop/imagesTr/{name}_0000.nii.gz'
    mask_path = f'/projects/pet_ct_challenge/l.cai/preprocess/Dataset104_Task1_crop/labelsTr/{name}.nii.gz'

    preprocessor_config_path='/projects/pet_ct_challenge/survival_code/json/preprocessor_config_task3.json'
    MODEL_DIR = "/projects/pet_ct_challenge/l.cai/task3_experiments/clinical_model_5fold_output_task3"#""'/projects/pet_ct_challenge/survival_code/clinical_model_5fold_output_task3'
    # model_dir_image='/projects/pet_ct_challenge/survival_code/image_model_5fold_output'
    preprocessed_data, input_size = preprocess_clinical_data_for_inference_docker(
                        json_path=json_path,
                        pet_path=pet_path,
                        mask_path=mask_path,
                        preprocessor_config_path=preprocessor_config_path
                    )
    print(f"Model input size determined to be: {input_size}")
    print(preprocessed_data)
    N_SPLITS = 10
    pet_data = load_nifti(pet_path)
    ct_data = load_nifti(ct_path)
    mask_data = load_nifti(mask_path)
    PATCH_SIZE = (96, 96, 96)
    # 1. 找到裁剪中心
    centroid = find_lesion_centroid(mask_data)

    # 2. 以该中心裁剪所有3D图像
    pet_patch = crop_around_center(pet_data, centroid, PATCH_SIZE)
    ct_patch = crop_around_center(ct_data, centroid, PATCH_SIZE)
    mask_patch = crop_around_center(mask_data, centroid, PATCH_SIZE)

    # 3. 对CT patch进行预处理
    ct_patch_preprocessed = preprocess_ct(ct_patch)

    # 4. 堆叠成3通道 (C, D, H, W) 的numpy数组
    #    通道顺序: CT, PET, Mask
    combined_patch = np.stack([
        ct_patch_preprocessed,
        pet_patch,
        mask_patch
    ], axis=0).astype(np.float32)
    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, 'best_model_weights_fold_*.pth')))
    inference_model = InferenceModel(model_paths=model_files, device=device)
    final_predict = inference_model.predict(combined_patch, preprocessed_data)

    filename = os.path.splitext(os.path.basename(json_path))[0]  # 先取文件名，再去掉后缀
    print(f"[{filename}] final_prediction to be: {final_predict}")
    # 确保目录存在
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    # 保存纯数值到 JSON 文件
    with open(json_output_path, 'w') as f:
        json.dump(final_predict['pred'], f)
    print(f"Saved final_predict {final_predict['pred']} to {json_output_path}")



























#
# #存储成json 做整体的测试  上面的是只有一个文件的
# def find_file_by_id(patient_id, directories, suffix=""):
#     """
#     在一个或多个目录中根据PatientID和后缀查找文件。
#     处理 CHUM-001 (ID) 与 CHUM-001000.nii.gz (文件名) 的匹配问题。
#     """
#     if not isinstance(directories, list):
#         directories = [directories]
#
#     for directory in directories:
#         if not os.path.isdir(directory):
#             continue
#         for filename in os.listdir(directory):
#             # 检查文件名是否以PatientID开头并以指定后缀结尾
#             if filename.startswith(patient_id) and filename.endswith(suffix):
#                 return os.path.join(directory, filename)
#     return None
#
#
# if __name__ == '__main__':
#
#
#     preprocessor_config_path='/projects/pet_ct_challenge/survival_code/json/preprocessor_config_task3.json'
#     MODEL_DIR = '/projects/pet_ct_challenge/survival_code/clinical_model_5fold_output_task3'
#     # model_dir_image='/projects/pet_ct_challenge/survival_code/image_model_5fold_output'
#
#     # 你的 JSON 文件目录
#     input_json_dir = '/projects/pet_ct_challenge/survival_code/json/input_task3'
#     LABEL_DIR = '/projects/pet_ct_challenge/task2/label_All'  # <--- 修改这里
#
#     # 3. PET原始图像所在的文件夹 (代码会自动在两个文件夹中搜索)
#     PET_DIR_1 = '/projects/pet_ct_challenge/l.cai/preprocess/Dataset104_Task1_crop/imagesTr'
#     PET_DIR_2 = '/projects/pet_ct_challenge/l.cai/preprocess/Task2/imagesTr'
#     # 遍历文件夹下所有 .json 文件
#     for filename in os.listdir(input_json_dir):
#         if filename.endswith('.json'):
#             json_path = os.path.join(input_json_dir, filename)
#
#             # 示例调用（pet_path 和 mask_path 你之后填）
#             patient_id = os.path.splitext(filename)[0]
#             mask_path = find_file_by_id(patient_id, [LABEL_DIR], suffix=".nii.gz")
#             pet_path = find_file_by_id(patient_id, [PET_DIR_1, PET_DIR_2], suffix="_0001.nii.gz")
#             ct_path = find_file_by_id(patient_id, [PET_DIR_1, PET_DIR_2], suffix="_0000.nii.gz")
#
#         preprocessed_data, input_size = preprocess_clinical_data_for_inference_docker(
#             json_path=json_path,
#             pet_path=pet_path,
#             mask_path=mask_path,
#             preprocessor_config_path=preprocessor_config_path
#         )
#         print(f"Model input size determined to be: {input_size}")
#         print(preprocessed_data)
#         N_SPLITS = 10
#         pet_data = load_nifti(pet_path)
#         ct_data = load_nifti(ct_path)
#         mask_data = load_nifti(mask_path)
#         PATCH_SIZE = (96, 96, 96)
#         # 1. 找到裁剪中心
#         centroid = find_lesion_centroid(mask_data)
#
#         # 2. 以该中心裁剪所有3D图像
#         pet_patch = crop_around_center(pet_data, centroid, PATCH_SIZE)
#         ct_patch = crop_around_center(ct_data, centroid, PATCH_SIZE)
#         mask_patch = crop_around_center(mask_data, centroid, PATCH_SIZE)
#
#         # 3. 对CT patch进行预处理
#         ct_patch_preprocessed = preprocess_ct(ct_patch)
#
#         # 4. 堆叠成3通道 (C, D, H, W) 的numpy数组
#         #    通道顺序: CT, PET, Mask
#         combined_patch = np.stack([
#             ct_patch_preprocessed,
#             pet_patch,
#             mask_patch
#         ], axis=0).astype(np.float32)
#         model_files = sorted(glob.glob(os.path.join(MODEL_DIR, 'best_model_weights_fold_*.pth')))
#         inference_model = InferenceModel(model_paths=model_files, device=device)
#         final_predict = inference_model.predict(combined_patch, preprocessed_data)
#
#         filename = os.path.splitext(os.path.basename(json_path))[0]  # 先取文件名，再去掉后缀
#         print(f"[{filename}] final_prediction to be: {final_predict}")




















# 一些测试
# def preprocess_clinical_data_for_inference(dataframe: pd.DataFrame, all_training_data: pd.DataFrame):
#     """
#     为推理准备临床数据。
#     使用在 *全部训练数据* 上拟合的 scaler 和 one-hot编码列。
#     """
#     CONTINUOUS_FEATURES = ["Age", "MTV", "NTV", "N", "T_SUV", "N_SUV", "TLG", "NLG"]
#     CATEGORICAL_FEATURES = [
#         "Gender", "Tobacco Consumption", "Alcohol Consumption",
#         "Performance Status", "M-stage", "Treatment"
#     ]
#
#     # 1. 从完整的训练数据中学习预处理规则
#     # Fit scaler on all training data
#     continuous_scaler = StandardScaler()
#     continuous_scaler.fit(
#         all_training_data[CONTINUOUS_FEATURES].fillna(all_training_data[CONTINUOUS_FEATURES].median()))
#
#     # Get all possible one-hot columns from training data
#     categorical_df_train = all_training_data[CATEGORICAL_FEATURES].astype(str).fillna('Unknown')
#     train_dummies = pd.get_dummies(categorical_df_train, prefix=CATEGORICAL_FEATURES, dtype=np.float32)
#     all_feature_columns = list(CONTINUOUS_FEATURES) + list(train_dummies.columns)
#
#     # 2. 将学习到的规则应用到新的推理数据上
#     # Handle continuous features for the new data
#     continuous_df_new = dataframe[CONTINUOUS_FEATURES].copy()
#     continuous_df_new = continuous_df_new.fillna(
#         all_training_data[CONTINUOUS_FEATURES].median())  # Use median from training data
#     continuous_scaled_array = continuous_scaler.transform(continuous_df_new)
#     continuous_scaled_df_new = pd.DataFrame(continuous_scaled_array, columns=CONTINUOUS_FEATURES, index=dataframe.index)
#
#     # Handle categorical features for the new data
#     categorical_df_new = dataframe[CATEGORICAL_FEATURES].copy().astype(str).fillna('Unknown')
#     categorical_encoded_df_new = pd.get_dummies(categorical_df_new, prefix=CATEGORICAL_FEATURES, dtype=np.float32)
#
#     # Combine and align columns to match training feature set
#     final_df_new = pd.concat([continuous_scaled_df_new, categorical_encoded_df_new], axis=1)
#     # Reindex to ensure all columns from training are present, filling missing ones with 0
#     final_df_aligned = final_df_new.reindex(columns=all_feature_columns, fill_value=0)
#
#     return final_df_aligned.values.astype(np.float32), len(all_feature_columns)
#
# full_training_dataframe = pd.read_csv('/projects/pet_ct_challenge/survival_code/HECKTOR_2025_Training_with_Radiomics_Features_task3.csv')
# preprocessed_data, _ = preprocess_clinical_data_for_inference(full_training_dataframe, full_training_dataframe)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
#
# # 如果 preprocessed_data 是 numpy 数组，先转换为 DataFrame（加列名可选）
# preprocessed_df = pd.DataFrame(preprocessed_data)
#
# print(preprocessed_df)
#
#
#
# import os
# def find_file_by_id(patient_id, directories, suffix=""):
#     """
#     在一个或多个目录中根据PatientID和后缀查找文件。
#     处理 CHUM-001 (ID) 与 CHUM-001000.nii.gz (文件名) 的匹配问题。
#     """
#     if not isinstance(directories, list):
#         directories = [directories]
#
#     for directory in directories:
#         if not os.path.isdir(directory):
#             continue
#         for filename in os.listdir(directory):
#             # 检查文件名是否以PatientID开头并以指定后缀结尾
#             if filename.startswith(patient_id) and filename.endswith(suffix):
#                 return os.path.join(directory, filename)
#     return None
# # 你的 JSON 文件目录
# input_json_dir = '/projects/pet_ct_challenge/survival_code/json/input_task3'
# LABEL_DIR = '/projects/pet_ct_challenge/task2/label_All'  # <--- 修改这里
#
# # 3. PET原始图像所在的文件夹 (代码会自动在两个文件夹中搜索)
# PET_DIR_1 = '/projects/pet_ct_challenge/l.cai/preprocess/Dataset104_Task1_crop/imagesTr'
# PET_DIR_2 = '/projects/pet_ct_challenge/l.cai/preprocess/Task2/imagesTr'
# # 遍历文件夹下所有 .json 文件
# for filename in os.listdir(input_json_dir):
#     if filename.endswith('.json'):
#         json_path = os.path.join(input_json_dir, filename)
#
#         # 示例调用（pet_path 和 mask_path 你之后填）
#         patient_id=os.path.splitext(filename)[0]
#         mask_path = find_file_by_id(patient_id, [LABEL_DIR], suffix=".nii.gz")
#         pet_path = find_file_by_id(patient_id, [PET_DIR_1, PET_DIR_2], suffix="_0001.nii.gz")
#         try:
#             features, dim = preprocess_clinical_data_for_inference_docker(
#                 json_path=json_path,
#                 pet_path=pet_path,
#                 mask_path=mask_path,
#                 preprocessor_config_path='/projects/pet_ct_challenge/survival_code/json/preprocessor_config_task3.json'
#             )
#
#             print(f"✔ 成功处理: {filename} → 特征维度: {dim}")
#             print(features)
#
#         except Exception as e:
#             print(f"❌ 处理失败: {filename} → 错误: {e}")




