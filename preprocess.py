import SimpleITK as sitk
import numpy as np
import torch
import warnings
from skimage.measure import label
import os

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    NormalizeIntensityd,
    # Sigmoid activation is included
    Activationsd,
)
from monai.data import MetaTensor



def sitk_to_metatensor(img_sitk: sitk.Image) -> MetaTensor:
    """
    SimpleITK ➜ MetaTensor, channel-first [1, Z, Y, X],
    *no* manual transpose.  Direction/spacing/origin preserved.
    """
    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # [Z, Y, X]
    arr = arr[None, ...]                                       # [1, Z, Y, X]

    spacing   = img_sitk.GetSpacing()       # (sx, sy, sz)
    origin    = img_sitk.GetOrigin()
    direction = img_sitk.GetDirection()     # 9-tuple row-major

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.reshape(direction, (3, 3)) * spacing
    affine[:3,  3] = origin

    meta = {
        "spacing":   spacing,
        "origin":    origin,
        "direction": direction,
        "affine":    affine,
    }
    return MetaTensor(arr, meta=meta)

 
def get_bounding_boxes(ct_sitk, pet_sitk):
    """
    Get the bounding boxes of the CT and PET images.
    This works since all images have the same direction.
    """
    ct_origin = np.array(ct_sitk.GetOrigin())
    pet_origin = np.array(pet_sitk.GetOrigin())

    # print("CT Origin:", ct_origin)
    # print("PET Origin:", pet_origin)
 
    ct_position_max = ct_origin + np.array(ct_sitk.GetSize()) * np.array(ct_sitk.GetSpacing())
    pet_position_max = pet_origin + np.array(pet_sitk.GetSize()) * np.array(pet_sitk.GetSpacing())

    # if np.any(bb_end - bb_start <= 0):
    #     raise ValueError("CT and PET images do not overlap in physical space!")
    
    return np.concatenate([
        np.maximum(ct_origin, pet_origin),
        np.minimum(ct_position_max, pet_position_max),
    ], axis=0)
 
def resample_images(ct_path, pet_path, mask_path=None):
    """
    Resample CT and PET images to specified resolution using SimpleITK.
    
    Args:
        ct_array: CT image as numpy array
        pet_array: PET image as numpy array
        
    Returns:
        Tuple of (resampled_ct_array, resampled_pet_array)
    """
    resampling = [1, 1, 1]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)
    if isinstance(ct_path, str):
        ct = sitk.ReadImage(ct_path)
    else:
        ct = ct_path
    
    if isinstance(pet_path, str):
        pt = sitk.ReadImage(pet_path)
    else:
        pt = pet_path
    # ct = sitk.Cast(ct, sitk.sitkFloat32)
    # pt = sitk.Cast(pt, sitk.sitkFloat32)
    bb = get_bounding_boxes(ct, pt)
    size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
    resampler.SetOutputOrigin(bb[:3])
    # print("type(size) =", type(size))
    # print("element types =", [type(k) for k in size])
    resampler.SetSize([int(k) for k in size])  
    #resampler.SetSize([int(max(0, x)) for x in size.tolist()])
    resampler.SetInterpolator(sitk.sitkBSpline)
    ct = resampler.Execute(ct)
    pt = resampler.Execute(pt)
    if mask_path is not None:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask = sitk.ReadImage(mask_path)
        mask = resampler.Execute(mask)
        return ct, pt, mask, bb
    else:
        # If no mask provided, just return ct and pt
        return ct,pt, bb



import SimpleITK as sitk

def register_pet_to_ct(ct, pet):
    # Read images if paths are given
    if isinstance(ct, str):
        ct = sitk.ReadImage(ct)
    if isinstance(pet, str):
        pet = sitk.ReadImage(pet)

    # Ensure both are 3D float images
    ct = sitk.Cast(ct, sitk.sitkFloat32)
    pet = sitk.Cast(pet, sitk.sitkFloat32)
    fixed_img = ct
    moving_img = pet
    # Set up registration
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
    #do transformation
    #ct_reg = sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID())
    pt_reg = sitk.Resample(pet, fixed_img, final_transform, sitk.sitkLinear, 0.0, pet.GetPixelID())
    #mask_reg = sitk.Resample(mask, fixed_img, final_transform, sitk.sitkNearestNeighbor, 0.0, mask.GetPixelID())

    return pt_reg



    
def get_roi_center(pet_tensor, z_top_fraction=0.75, z_score_threshold=1.0):
    """
    Calculates the center of the largest high-intensity region in the top part of the PET scan.
    """
    # 1. Isolate top of the scan based on the z-axis
    image_shape_voxels = np.array(pet_tensor.shape)
    crop_z_start = int(z_top_fraction * image_shape_voxels[2])
    top_of_scan = pet_tensor[..., crop_z_start:]
 
    # 2. Threshold to find high-intensity regions (potential brain/tumor)
    # Using a small epsilon to avoid division by zero in blank images
    mask = ((top_of_scan - top_of_scan.mean()) / (top_of_scan.std() + 1e-8)) > z_score_threshold
    
    if not mask.any():
        # If no pixels are above the threshold, fall back to the geometric center of the top part
        warnings.warn("No high-intensity region found. Using geometric center of the upper scan region.")
        center_in_top = (np.array(top_of_scan.shape) / 2).astype(int)
    else:
        # Find the largest connected component to remove noise
        labeled_mask, num_features = label(mask, return_num=True, connectivity=3)
        if num_features > 0:
            component_sizes = np.bincount(labeled_mask.ravel())[1:]  # ignore background
            largest_component_label = np.argmax(component_sizes) + 1
            largest_component_mask = labeled_mask == largest_component_label
            comp_idx = np.argwhere(largest_component_mask)
        else:  # Should not happen if mask.any() is true, but as a safeguard
            comp_idx = np.argwhere(mask)
 
        # 3. Calculate the centroid of the largest component
        center_in_top = np.mean(comp_idx, axis=0)
 
    # 4. Adjust center to be in the original full-image coordinate system
    center_full_image = center_in_top + np.array([0, 0, crop_z_start])
    return center_full_image.astype(int)
 
def crop_neck_region_sitk(
        ct_sitk:  sitk.Image,
        pet_sitk: sitk.Image,
        mask_sitk: sitk.Image = None,
        crop_box_size=(200, 200, 310),
        z_top_fraction=0.75,
        z_score_threshold=1.0,
):
    # ------------------------------------------------------------------
    # 1. Convert PET to numpy for ROI-finding (SimpleITK gives z,y,x order)
    # ------------------------------------------------------------------
    pet_np_zyx = sitk.GetArrayFromImage(pet_sitk)           # [z, y, x]
    pet_np_xyz = np.transpose(pet_np_zyx, (2, 1, 0))       # [x, y, z]
    pet_tensor = torch.from_numpy(pet_np_xyz).float()

    # ------------------------------------------------------------------
    # 2. Determine the crop centre and bounding box in voxel coordinates
    # ------------------------------------------------------------------
    crop_box_size = np.asarray(crop_box_size, dtype=int)
    center = get_roi_center(pet_tensor,
                            z_top_fraction=z_top_fraction,
                            z_score_threshold=z_score_threshold)

    img_shape = np.asarray(pet_np_xyz.shape)
    box_start = np.clip(center - crop_box_size // 2, 0, img_shape)
    box_end   = np.clip(box_start + crop_box_size, 0, img_shape)

    # Guard in case image is smaller than requested box
    box_start = np.maximum(box_end - crop_box_size, 0)

    # SimpleITK wants index & size in (x, y, z) order
    index = [int(i) for i in box_start]
    size  = [int(e - s) for s, e in zip(box_start, box_end)]

    # ------------------------------------------------------------------
    # 3. Crop with RegionOfInterest (origin adjusted automatically)
    # ------------------------------------------------------------------
    if mask_sitk is not None:
        ct_crop  = sitk.RegionOfInterest(ct_sitk,  size=size, index=index)
        et_crop = sitk.RegionOfInterest(pet_sitk, size=size, index=index)
        mask_crop = sitk.RegionOfInterest(mask_sitk, size=size, index=index)
        return ct_crop, et_crop, mask_crop, box_start, box_end
    else: 
        ct_crop  = sitk.RegionOfInterest(ct_sitk,  size=size, index=index)
        pet_crop = sitk.RegionOfInterest(pet_sitk, size=size, index=index)

        return ct_crop, pet_crop, box_start, box_end

def get_preprocessing_transforms(keys, final_size=(200, 200, 310)):
    """
    Defines the sequence of deterministic transforms to be applied to each case.
    This version includes Sigmoid activation for CT and PET.
    
    Args:
        keys (list): List of keys ('ct', 'pet', 'label') to apply transforms to.
        final_size (tuple): The final spatial size to pad the images to after cropping.
        
    Returns:
        monai.transforms.Compose: The composition of all preprocessing transforms.
    """
    return Compose([
        # 3. Reorient all images to a standard 'RAS' orientation
        Orientationd(keys=keys, axcodes="RAS"),
        
        # 4. Normalize images and apply sigmoid
        # 4a. Re-scale CT intensity to [0, 1] range.
        ScaleIntensityRanged(
            keys=["ct"], a_min=-250, a_max=250, b_min=-6.0, b_max=6.0, clip=True
        ),
        # 4b. Normalize PET to zero mean, unit variance.
        NormalizeIntensityd(keys=["pet"], nonzero=True, channel_wise=True),
        
        # # =========================================================================
        # # CONFIRMATION: Sigmoid is applied here to CT and PET as a soft clamp.
        # # =========================================================================
        # Activationsd(keys=["ct", "pet"], sigmoid=True),
        
        # 5. Crop away empty background based on CT and then pad all to a uniform size.
        CropForegroundd(keys=keys, source_key="ct", allow_smaller=True),
        SpatialPadd(keys=keys, spatial_size=final_size, method="end"),
    ])

def apply_monai_transforms(ct_sitk: sitk.Image,
                            pt_sitk: sitk.Image,
                            mask_sitk: sitk.Image = None,
                            final_size = (310, 200, 200)):
    """
    Run the deterministic MONAI preprocessing on in-memory SimpleITK volumes.

    Parameters
    ----------
    ct_sitk, pt_sitk : sitk.Image
        Aligned, resampled CT / PET volumes.
    final_size       : tuple[int, int, int]
        Target padded size (passed through to get_preprocessing_transforms).

    Returns
    -------
    ct_t, pet_t : monai.data.MetaTensor   (shape = [C, H, W, D])
    meta        : dict                    (spacing, origin, direction, …)
    """
    # ------------------------------------------------------------------
    # 1. Wrap into MetaTensors so MONAI keeps spatial metadata
    # ------------------------------------------------------------------
    ct_mt  = sitk_to_metatensor(ct_sitk)

    pet_mt = sitk_to_metatensor(pt_sitk)
    data = {"ct": ct_mt, "pet": pet_mt}


    # ------------------------------------------------------------------
    # 2. Build the preprocessing Compose 
    # ------------------------------------------------------------------
    
    #out    = xforms(data)
    # ------------------------------------------------------------------
    # 3. Execute transforms
    # ------------------------------------------------------------------
    if mask_sitk is not None:
        mask_mt = sitk_to_metatensor(mask_sitk)
        data["label"] = mask_mt
        xforms  = get_preprocessing_transforms(keys=["ct", "pet", "label"],
                                                 final_size=final_size)
        out = xforms(data)
        ct_proc  = out["ct"]     # MetaTensor, 1×H×W×D
        pet_proc = out["pet"]
        mask_proc = out["label"]
        meta = ct_proc.meta 
        return ct_proc, pet_proc, mask_proc, meta 
    else: 
        xforms  = get_preprocessing_transforms(keys=["ct", "pet"],
                                                 final_size=final_size)
        out    = xforms(data)
        ct_proc  = out["ct"]     # MetaTensor, 1×H×W×D
        pet_proc = out["pet"]
        meta = ct_proc.meta               # <— this is the live dict


        return ct_proc, pet_proc, meta   




folder = "/data/groups/beets-tan/archive/HECKTOR 2025 Training Data/Task 1/"
output_dir = "/processing/l.cai/imagesTr_resampled_cropped_npy"
output_label_dir = "/processing/l.cai/labelsTr_resampled_cropped_npy"

folder_task2 = "/data/groups/beets-tan/archive/HECKTOR 2025 Training Data/Task 2/"


task1_files = os.listdir(folder)
task2_files = os.listdir(folder_task2)

only_task2_files = [case for case in task2_files if case not in task1_files]


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_label_dir):
    os.makedirs(output_label_dir)


print(len(only_task2_files), len(task1_files))

for case in sorted(task1_files)[:]:#os.listdir(folder):
    print(case)
    # if not (case == "MDA-410" or case == "MDA-445"):
    #     continue
    
    case_path = os.path.join(folder, case)
    if os.path.isdir(case_path):
        ct = os.path.join(case_path, f"{case}__CT.nii.gz")
        pt = os.path.join(case_path, f"{case}__PT.nii.gz")
        mask = os.path.join(case_path, f"{case}.nii.gz")
        if not os.path.exists(mask):
            mask = None
            try:
                ct, pt,  bb = resample_images(ct, pt)
            except:
            #if True:
                pt = register_pet_to_ct(ct, pt)
                ct, pt,  bb = resample_images(ct, pt)
                print(ct.GetSize(), pt.GetSize())
            ct_cropped, pet_cropped,  box_start, box_end = crop_neck_region_sitk(ct, pt)
        
            ct_proc, pt_proc = ct_cropped, pet_cropped

            print( ct_proc.GetSize(), pt_proc.GetSize())
        else:
            ct, pt, mask, bb = resample_images(ct, pt, mask)
            ct_cropped, pet_cropped, mask_cropped, box_start, box_end = crop_neck_region_sitk(ct, pt, mask)
            
            ct_proc, pt_proc, mask_proc = ct_cropped, pet_cropped, mask_cropped

            print(np.unique(mask_proc), ct_proc.GetSize(), pt_proc.GetSize(), mask_proc.GetSize())

        ###for other users
        # ct_proc = sitk.GetImageFromArray(ct_proc)
        # pt_proc = sitk.GetImageFromArray(pt_proc)
        # mask_proc = sitk.GetImageFromArray(mask_proc)
        #ct_proc = sitk.GetImageFromArray(np.squeeze(ct_proc, 0))  # [Z, Y, X]
        #pt_proc = sitk.GetImageFromArray(np.squeeze(pt_proc, 0))  # [Z, Y, X]
        #mask_proc = sitk.GetImageFromArray(np.squeeze(mask_proc, 0))  # [Z, Y, X]

        sitk.WriteImage(ct_proc, os.path.join("/projects/pet_ct_challenge/l.cai/preprocess/Dataset104_Task1_crop/imagesTr", f"{case}_0000.nii.gz"))
        sitk.WriteImage(pt_proc, os.path.join("/projects/pet_ct_challenge/l.cai/preprocess/Dataset104_Task1_crop/imagesTr", f"{case}_0001.nii.gz"))
        if not mask_proc == None:
            sitk.WriteImage(mask_proc, os.path.join("/projects/pet_ct_challenge/l.cai/preprocess/Dataset104_Task1_crop/labelsTr", f"{case}.nii.gz"))


        

       
    


