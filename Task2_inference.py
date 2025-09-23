from pathlib import Path
import json
from glob import glob
import SimpleITK as sitk
import numpy
import os
import numpy as np
import sys
import subprocess
import warnings
import torch
warnings.filterwarnings("ignore")
from tqdm import tqdm
import sys
from functools import partial
import shutil
from resources.preprocess import resample_images, crop_neck_region_sitk, apply_monai_transforms
#from resources.utils import load_model_from_checkpoint, arrays_to_tensor, run_inference
from resources.postprocess import prediction_to_original_space
import torch.nn as nn
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
#import glob
# tqdm = partial(tqdm, file=sys.stdout)
# os.environ["TQDM_DISABLE"] = "1"
def write_json_file(*, location, content):
    """Writes a json file."""
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

def nrrd_reg_rigid_ref(ct, pt): #ref_path, moving_img, pt):
    # if ref_path == "":
    #     fixed_img = moving_img
    # else:
    #     fixed_img = sitk.ReadImage(ref_path)
    #fixed_img = ref_path
    fixed_img = ct
    moving_img = ct



    transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

    # multi-resolution rigid registration using Mutual Information
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
        )

    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)
    final_transform = registration_method.Execute(fixed_img, moving_img)
    #print(final_transform)
    #do transformation
    #ct_reg = sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID())
    pt_reg = sitk.Resample(pt, fixed_img, final_transform, sitk.sitkLinear, 0.0, pt.GetPixelID())
    #mask_reg = sitk.Resample(mask, fixed_img, final_transform, sitk.sitkNearestNeighbor, 0.0, mask.GetPixelID())
    return pt_reg#ct_reg, pt_reg, mask_reg, final_transform


def write_nifti_to_mha(input_file, output_file):
    """
    Convert a NIfTI file to MHA format using SimpleITK.
    """
    # Read the NIfTI file
    image = input_file#sitk.ReadImage(input_file)

    # Write the image to MHA format
    sitk.WriteImage(image, output_file, useCompression=True)
    print(f"Converted {input_file} to {output_file}")

def predict(task, nnunet_results, nnunet_inp_dir, nnunet_out_dir,trainer="STUNetTrainer_small", network="3d_fullres",
                checkpoint="model_final_checkpoint", folds="0,1,2,3,4", store_probability_maps=False,
                disable_augmentation=False, disable_patch_overlap=False):
        """
        Use trained nnUNet network to generate segmentation masks
        """

        # Set environment variables

        #os.environ["nnUNet_preprocessed"] = str(nnunet_results)
        #os.environ["nnUNet_raw"] = str(nnunet_results)

        os.environ['nnUNet_results'] = str(nnunet_results)

        # Run prediction script
        cmd = [
            'nnUNetv2_predict',
            '-d', task,
            '-i', str(nnunet_inp_dir),
            '-o', str(nnunet_out_dir),
            "-c", "3d_fullres",
            '-tr', trainer,
            # '--num_threads_preprocessing', '2',
            # '--num_threads_nifti_save', '1'
        ]

        if folds:
            cmd.append('-f')
            cmd.extend(folds.split(','))

        if checkpoint:
            cmd.append('-chk')
            cmd.append(checkpoint)

        # if store_probability_maps:
        #     cmd.append('--save_npz')

        # if disable_augmentation:
        #     cmd.append('--disable_tta')

        # if disable_patch_overlap:
        #     cmd.extend(['--step_size', '1'])

        subprocess.check_call(cmd)

# def resample_image(image, new_spacing, interpolator=sitk.sitkLinear):
#     original_spacing = image.GetSpacing()
#     original_size = image.GetSize()
#     new_size = [
#         int(round(osz * ospc / nspc))
#         for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
#     ]
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetOutputSpacing(new_spacing)
#     resampler.SetSize(new_size)
#     resampler.SetInterpolator(interpolator)
#     resampler.SetOutputDirection(image.GetDirection())
#     resampler.SetOutputOrigin(image.GetOrigin())
#     resampler.SetDefaultPixelValue(0)
#     return resampler.Execute(image)


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.nii.gz"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    print(f"Found {len(input_files)} files in {location}: {input_files}")
    return input_files[0]


def write_image(*, location, image, filename="output.mha"):
    location.mkdir(parents=True, exist_ok=True)

    if isinstance(image, sitk.Image):
        img = image  # already SimpleITK
    else:  # assume NumPy
        img = sitk.GetImageFromArray(image)

    sitk.WriteImage(img, str(location / filename), useCompression=True)



def write_array_as_image_file(*, location, array, filename="output.mha"):
    location.mkdir(parents=True, exist_ok=True)

    if isinstance(array, sitk.Image):
        img = array                       # already SimpleITK
    else:                                 # assume NumPy
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]              # drop batch dim if present
        img = sitk.GetImageFromArray(array)

    sitk.WriteImage(img, str(location / filename), useCompression=True)





def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

# ==============================================================================
# 0. 从训练脚本中复用必要的定义
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- 复用模型定义 ---
# ==============================================================================
# 2. 新的模型定义
# ==============================================================================
class MaskBranch(nn.Module):
    """
    一个极简的3D CNN，专门用于处理Mask输入。
    它只包含一个卷积层，用于将Mask转换为与主干网络匹配的特征图。
    """
    def __init__(self, in_channels=2, out_channels=64):
        super(MaskBranch, self).__init__()
        # 目标：将 (B, 2, 96, 96, 96) 的Mask处理成 (B, 64, 96, 96, 96) 的特征图
        # 我们使用一个卷积层来完成通道扩展和特征学习
        # kernel_size=3, stride=1, padding=1 的组合可以保持空间尺寸不变
        self.branch = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.branch(x)
class DualBranchFusionResNet(nn.Module):
    """
    图像(CT+PET+Mask) + 临床特征联合模型
    - 图像：CT/PET 进入 MONAI ResNet；Mask -> MaskBranch；在 conv1+bn+act 后相加融合
    - 临床：简单 MLP 处理；与图像全局池化后的向量拼接；分类头输出离散生存时间bins的logits
    """
    def __init__(self, num_classes=4, clinical_input_size=128, dropout_p=0.1):
        super().__init__()

        # 1) 图像主干（MONAI 1.5.0 用 n_input_channels）
        self.image_backbone = ResNet(
            block="basic", layers=[2, 2, 2, 2], block_inplanes=[32, 64, 128, 256],
            spatial_dims=3, n_input_channels=2, num_classes=num_classes,
            act=Act.PRELU, norm="batch"
        )
        # conv1 输出通道数，用于对齐 mask 分支
        self.img_c1_out = self.image_backbone.conv1.out_channels

        # 2) Mask 分支：输入 one-hot 的肿瘤/淋巴结两通道
        self.mask_branch = MaskBranch(in_channels=2, out_channels=self.img_c1_out)

        # 3) 去掉原来的 fc，保留 image feature 向量
        self.image_feat_dim = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity()

        # 4) 临床特征处理分支（可替代你原 SimpleNN 的前两层）
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.BatchNorm1d(32),
        )
        self.clin_feat_dim = 128

        # 5) 分类头：拼接(图像特征 + 临床特征)后输出 num_classes
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.image_feat_dim + self.clin_feat_dim),
            nn.Linear(self.image_feat_dim + self.clin_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, clinical):
        """
        x: [B, 3, D, H, W]  (前2通道=CT+PET，第3通道=mask整数标签(0/1/2))
        clinical: [B, clinical_input_size]
        """
        # --- 图像分支 ---
        image_tensor = x[:, :2, ...]          # [B,2,D,H,W]
        mask_tensor_int = x[:, 2:3, ...].long()  # [B,1,D,H,W]

        # mask -> one-hot 后取(肿瘤/淋巴结)二通道
        mask_one_hot = torch.nn.functional.one_hot(mask_tensor_int.squeeze(1), num_classes=3).permute(0, 4, 1, 2,
                                                                                                      3).float()
        lesion_mask = mask_one_hot[:, 1:, ...]                               # [B,2,D,H,W]

        # mask 特征
        mask_features = self.mask_branch(lesion_mask)                        # [B,C1,d?,h?,w?]

        # 图像 conv1+bn+act
        img = self.image_backbone.conv1(image_tensor)
        img = self.image_backbone.bn1(img)
        img = self.image_backbone.act(img)                                   # [B,C1,D',H',W']

        # 对齐空间尺寸再融合
        if mask_features.shape[2:] != img.shape[2:]:
            mask_features = F.interpolate(mask_features, size=img.shape[2:], mode="trilinear", align_corners=False)

        fused = img + mask_features

        # 后续残块
        x = self.image_backbone.maxpool(fused)
        x = self.image_backbone.layer1(x)
        x = self.image_backbone.layer2(x)
        x = self.image_backbone.layer3(x)
        x = self.image_backbone.layer4(x)
        x = self.image_backbone.avgpool(x)
        x = torch.flatten(x, 1)  # [B, image_feat_dim]

        # --- 临床分支 ---
        c = self.clinical_processor(clinical)  # [B, clin_feat_dim]

        # --- 融合 + 分类 ---
        feat = torch.cat([x, c], dim=1)       # [B, image_feat_dim+clin_feat_dim]
        feat = self.dropout(feat)
        logits = self.classifier(feat)        # [B, num_classes] （离散时间 bins 的 logits）

        return logits


# 定义特征
CONTINUOUS_FEATURES = ["Age", "MTV", "NTV", "T_SUV", "N_SUV", "TLG", "NLG"]
CATEGORICAL_FEATURES = [
    "Gender", "Tobacco Consumption", "Alcohol Consumption",
    "Performance Status", "M-stage", "Treatment"
]



def build_and_save_preprocessors(
    csv_path='/projects/pet_ct_challenge/survival_code/HECKTOR_2025_Training_with_Radiomics_Features.csv',
    output_json_path='/projects/pet_ct_challenge/survival_code/json/preprocessor_config.json'
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


build_and_save_preprocessors()


def preprocess_clinical_data_for_inference_docker(json_path, pet_path, mask_path,
                                                 preprocessor_config_path='/projects/pet_ct_challenge/survival_code/json/preprocessor_config.json'):
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

        model = DualBranchFusionResNet(num_classes=4,clinical_input_size=input_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 设置为评估模式
        models.append(model)
        print(f"  -> Loaded model for fold {fold}")
    return models

# ==============================================================================
# 2. 主推理函数
# ==============================================================================
# --- 复用风险计算函数 ---
def _calculate_risk(h):
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1)
    return risk

def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata()
class InferenceModel:
    """
    Ensemble inference class that loads multiple fold models and combines predictions.
    """

    def __init__(self, model_paths, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_paths = model_paths
        self.models = self._load_models()
        print(f"Loaded {len(self.models)} models for ensemble inference.")

    def _load_models(self):
        """Loads all models from the specified paths."""
        loaded_models = []
        for path in self.model_paths:
            # Instantiate the model architecture
            model = DualBranchFusionResNet(num_classes=4,clinical_input_size=input_size).to(device)
            # Load the saved state dictionary
            model.load_state_dict(torch.load(path, map_location=self.device))
            # Set the model to evaluation mode
            model.eval()
            loaded_models.append(model)
        return loaded_models

    def predict(self, images,clinical):
        """
        Makes predictions using an ensemble of all loaded models.
        The logits from all models are averaged before calculating the final risk score.
        """
        all_ensemble_logits = []

        with torch.no_grad():
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()  # 转成Tensor并确保是float类型
                images=images.unsqueeze(0)
            images = images.to(self.device)
            clinical = torch.tensor(clinical, dtype=torch.float32).to(device)
            # Store predictions from each model in the ensemble
            fold_logits = []
            for model in self.models:
                outputs = model(images,clinical)
                fold_logits.append(outputs)

            # Average the logits across all models
            # Shape: [num_models, batch_size, num_classes] -> [batch_size, num_classes]
            ensemble_logits = torch.stack(fold_logits).mean(dim=0)

            all_ensemble_logits.append(ensemble_logits.cpu())
        # Concatenate all batch results
        final_logits = torch.cat(all_ensemble_logits, dim=0)

        # Calculate risk scores from the averaged logits
        risk_scores=_calculate_risk(final_logits)
        return risk_scores

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



# # Set nnUNet environment variables to point to your mounted directories inside Docker
# os.environ["nnUNet_raw_data_base"] = "/input"
# os.environ["nnUNet_preprocessed"] = "/preprocessed"
#os.environ["RESULTS_FOLDER"] = "/resources/nnUNet_results"

if __name__ == "__main__":
    INPUT_PATH = Path("/input")
    OUTPUT_PATH = Path("/output")
    RESOURCE_PATH = Path("resources")
    preprocessed = Path("/tmp")
    os.environ["RESULTS_FOLDER"] = "/resources/nnUNet_results"
    # input_folder = sys.argv[1]  # /input
    # output_folder = sys.argv[2] # /output
    #model_folder = os.path.join(os.environ["RESULTS_FOLDER"], "Dataset104_Task1_crop")
    #show_torch_cuda_info()
    
    try:
        ct_planning = load_image_file_as_array(
            location=INPUT_PATH / "images/ct-planning",
        )
    except (FileNotFoundError, IndexError):
        print(f"No image files found in {INPUT_PATH}/images/ct-planning")
        ct_planning = None
    
    try:
        rt_dose_map = load_image_file_as_array(
            location=INPUT_PATH / "images/rt-dose",
        )
    except (FileNotFoundError, IndexError):
        print(f"No image files found in {INPUT_PATH}/images/rt-dose")
        rt_dose_map = None
    
    
    

    ct_path  = load_image_file_as_array(
        location=INPUT_PATH / "images/ct",
    )
    input_electronic_health_record = load_json_file(
        location=INPUT_PATH / "ehr.json",
    )
    pt_path = load_image_file_as_array(
        location=INPUT_PATH / "images/pet",
    )

    print(f"CT image path: {ct_path}")
    print(f"PT image path: {pt_path}")

    try:
        ct, pt, bb= resample_images(
            ct_path=ct_path,
            pet_path=pt_path,
        )
    except: 
        pt = register_pet_to_ct(ct, pt)
        ct, pt,  bb = resample_images(ct, pt)


    ct = sitk.Cast(ct, sitk.sitkFloat32)
    pt = sitk.Cast(pt, sitk.sitkFloat32)
    print(ct.GetSize())
    print(pt.GetSize())


    #pt = nrrd_reg_rigid_ref(ct, pt)
    # crop the images to the bounding box
    ct_cropped, pet_cropped, box_start, box_end = crop_neck_region_sitk(
        ct_sitk=ct,
        pet_sitk=pt,
    )

    # apply transformation to the cropped images
    #ct_transformed, pet_transformed, meta = apply_monai_transforms(ct_cropped, pet_cropped)

    #target_spacing = (1.0, 1.0, 3.0)
    os.makedirs(preprocessed / "Task104_Task1_crop", exist_ok=True)
    os.makedirs(preprocessed / "Task104_Task1_crop/labelsTs", exist_ok=True)


    write_image(
        location=preprocessed / "Dataset104_Task1_crop/imagesTs",
        image=ct_cropped,
        filename=ct_path.split('/')[-1].split('.')[0]+"_0000.nii.gz"#[i.split('/')[-1] for i in ct_path]#"ct_resampled.mha",
    )

    write_image(
        location=preprocessed / "Dataset104_Task1_crop/imagesTs",
        image=pet_cropped,
        filename=ct_path.split('/')[-1].split('.')[0]+"_0001.nii.gz"#[i.split('/')[-1] for i in ct_path]#"ct_resampled.mha",
    )


    # Run nnUNet prediction
    predict("104", nnunet_results=RESOURCE_PATH / "nnUNet_results", nnunet_inp_dir=preprocessed / "Dataset104_Task1_crop"/"imagesTs",
              nnunet_out_dir=preprocessed / "Dataset104_Task1_crop"/"labelsTs", trainer="STUNetTrainer_small", network="3d_fullres",
              checkpoint="checkpoint_final.pth", folds="0,1,2,3,4", store_probability_maps=False,
              disable_augmentation=False, disable_patch_overlap=False)



    mask_path = load_image_file_as_array(
        location=preprocessed / "Dataset104_Task1_crop/labelsTs",
    )
    pet_path_cropped = preprocessed / "Dataset104_Task1_crop/imagesTs" / (ct_path.split('/')[-1].split('.')[0]+"_0001.nii.gz")
    pet_path = pet_path_cropped
    preprocessor_config_path = RESOURCE_PATH / 'survival_code/json/preprocessor_config.json'
    MODEL_DIR = RESOURCE_PATH / 'survival_code/clinical_model_5fold_output'
    model_dir_image= RESOURCE_PATH / 'survival_code/image_model_5fold_output'
    json_path = INPUT_PATH / "ehr.json"

    preprocessed_data, input_size = preprocess_clinical_data_for_inference_docker(
                        json_path=json_path,
                        pet_path=pet_path_cropped,
                        mask_path=mask_path,
                        preprocessor_config_path=preprocessor_config_path
                    )
    print(f"Model input size determined to be: {input_size}")
    print(preprocessed_data)

    N_SPLITS = 5

    ct_data = load_nifti(ct_path)
    pet_data = load_nifti(pet_path)
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

    # --- 计算4个MIP图像 ---

    # ==================== 关键修改点 ====================
    combined_patch = np.stack([
        ct_patch_preprocessed,
        pet_patch,
        mask_patch
    ], axis=0).astype(np.float32)
    model_files = sorted(glob.glob(os.path.join(model_dir_image, 'best_model_fold_*.pth')))
    inference_model = InferenceModel(model_paths=model_files, device=device)
    final_risk_scores_iamge = inference_model.predict(combined_patch,preprocessed_data)

    

   

    json_output_path = OUTPUT_PATH / "rfs.json"

    write_json_file(
        location=OUTPUT_PATH / "rfs.json", 
        content=float(final_risk_scores_iamge)
    )
    
    
    print(f"Saved average risk score {average_risk_score:.4f} to {json_output_path}")




























