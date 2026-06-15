import os
import glob
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.networks.nets import UNet
import scipy.ndimage as ndimage

# =========================
# CONFIGURATION
# =========================
INPUT_DIR = r"Data\Testing_data\imagesTs"
OUTPUT_DIR = r"Data\Inference_8{b}"

MODEL_WEIGHTS = "best_model_8{b}.pth"

NUM_CLASSES = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ===========================================================================
# PADDING FUNCTION (NO CROP) only till 512 if the image size is more than 512 
# then that constraint is not added)
# ===========================================================================
def pad_to_512(img):
    h, w = img.shape

    pad_h = max(0, 512 - h)
    pad_w = max(0, 512 - w)

    img = np.pad(
        img,
        ((pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2)),
        mode='constant'
    )

    return img

# =========================
# REMOVE PADDING FUNCTION
# =========================
def remove_padding(pred, original_shape):
    h, w = original_shape

    pad_h = pred.shape[0] - h
    pad_w = pred.shape[1] - w

    start_h = pad_h // 2
    start_w = pad_w // 2

    return pred[start_h:start_h + h, start_w:start_w + w]

# =========================
# LOAD MODEL
# =========================
print("Loading model...")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=NUM_CLASSES,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.eval()

print("Model loaded successfully")

# =========================
# LOAD FILES
# =========================
input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.nii*")))

print(f"Found {len(input_files)} volumes")

# =========================
# INFERENCE
# =========================
with torch.no_grad():

    for f_path in tqdm(input_files, desc="Processing"):

        fname = os.path.basename(f_path)

        if fname.endswith(".nii.gz"):
            out_name = fname[:-7] + ".nii"
        else:
            out_name = fname

        out_path = os.path.join(OUTPUT_DIR, out_name)

    # =========================
    # Load volume
    # =========================
        img_obj = nib.load(f_path)
        img_data = img_obj.get_fdata()
        affine = img_obj.affine

        original_shape = img_data.shape
#    =========================
    # Resample (same as training)
    # =========================
        spacing = img_obj.header.get_zooms()[:3]

        target_spacing = (1.0, 1.0, spacing[2])

        resize_factors = (
            spacing[0] / target_spacing[0],
            spacing[1] / target_spacing[1],
                1
        )

        img_data = ndimage.zoom(img_data, resize_factors, order=1)

    # =========================
    # Prepare output
    # =========================
        pred_volume_resampled = np.zeros(img_data.shape, dtype=np.uint8)

        depth = img_data.shape[-1]

    # =========================
    # Slice-wise inference
    # =========================
        for z in range(depth):

            slice_data = img_data[:, :, z]
            original_2d_shape = slice_data.shape
        # =========================
        # Padding ONLY (NO CROP)
        # =========================
            slice_padded = pad_to_512(slice_data)

            # =========================
            # Convert to tensor
            # =========================
            slice_tensor = torch.tensor(
                slice_padded,
                dtype=torch.float32
             ).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # =========================
            # Z-score normalization)
            # =========================
            # Ignore background (non-zero only)
            nonzero_pixels = slice_tensor[slice_tensor != 0]

            if nonzero_pixels.numel() > 0:
                mean = nonzero_pixels.mean()
                std = nonzero_pixels.std()

                if std > 1e-6:
                    slice_tensor = (slice_tensor - mean) / std
                    
            # =========================
            # Predict
            # =========================
            output = model(slice_tensor)
            pred_slice = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            # =========================
            # Remove padding
            # =========================
            pred_restored = remove_padding(pred_slice, original_2d_shape)

            pred_volume_resampled[:, :, z] = pred_restored

            # =========================
            # Restore original resolution
            # =========================
        pred_volume_original = ndimage.zoom(
            pred_volume_resampled,
            (
                original_shape[0] / pred_volume_resampled.shape[0],
                original_shape[1] / pred_volume_resampled.shape[1],
                1
            ),
            order=0
        )

            # =========================
            # Save output
            # =========================
        pred_nifti = nib.Nifti1Image(pred_volume_original, affine)
        nib.save(pred_nifti, out_path)

    print("\n✔ Inference complete!")
    print(f"Saved in: {OUTPUT_DIR}")