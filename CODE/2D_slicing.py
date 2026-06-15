import os
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
import scipy.ndimage as ndimage
from sklearn.model_selection import train_test_split

# =========================
# Configuration
# =========================
DATA_DIR = r"Data\Training_data"
TRAIN_IMAGES = os.path.join(DATA_DIR, "imagesTr")
TRAIN_LABELS = os.path.join(DATA_DIR, "labelsTr")

OUTPUT_DIR = r"Data\2d_slice_5{3}"

# Creating Paths
OUT_IMAGES_TRAIN = os.path.join(OUTPUT_DIR, "imagesTr")
OUT_LABELS_TRAIN = os.path.join(OUTPUT_DIR, "labelsTr")

OUT_IMAGES_VAL = os.path.join(OUTPUT_DIR, "imagesVal")
OUT_LABELS_VAL = os.path.join(OUTPUT_DIR, "labelsVal")

# Creating folders if doesn't exists
os.makedirs(OUT_IMAGES_TRAIN, exist_ok=True)
os.makedirs(OUT_LABELS_TRAIN, exist_ok=True)
os.makedirs(OUT_IMAGES_VAL, exist_ok=True)
os.makedirs(OUT_LABELS_VAL, exist_ok=True)

# =========================
# Defining Padding funcction 
# =========================
def pad_or_crop_512(img):
    h, w = img.shape

    # Pad if smaller
    pad_h = max(0, 512 - h) # Images greater than 512 then 0 padding otherwise 512 - h
    pad_w = max(0, 512 - w)

    img = np.pad(
        img,
        ((pad_h // 2, pad_h - pad_h // 2),
         (pad_w // 2, pad_w - pad_w // 2)), # ((top, bottom), (left, right)), to centre the image 
        mode='constant'
    )

    return img

# =========================
# Load volumes
# =========================
images = sorted(glob.glob(os.path.join(TRAIN_IMAGES, "*.nii*"))) # sorted because order of the image and label need to be consitsent 
labels = sorted(glob.glob(os.path.join(TRAIN_LABELS, "*.nii*")))

data_pairs = list(zip(images, labels))                              # [(image1, label1), (image2, label2),...]

# =========================
# Train/Val split (volume level)
# =========================
train_pairs, val_pairs = train_test_split(
    data_pairs, test_size=0.2, random_state=42
)  # Volume wise splitting is done, Train → Patient A, B, C, ..., Validation → Patient X, Y, Z, ..., this implies the model has neverseen the valodation patients before

print(f"Train volumes: {len(train_pairs)} | Val volumes: {len(val_pairs)}")

# =========================
# Processing function
# =========================
def process(pairs, out_img_dir, out_lbl_dir):
    total_slices = 0

    for img_path, lbl_path in tqdm(pairs): # iterates over (patient1_image, patient1_label), (patient2_image, patient2_label)

        base_name = os.path.basename(img_path).split('.')[0] # Saving the base name, Split at . e.g., patient_001.nii.gz → patient_001

        img_obj = nib.load(img_path) # This function opens .nii file and reads metadata + structure and returns an object(Nifti object)
        lbl_obj = nib.load(lbl_path)

        img_data = img_obj.get_fdata() # get_fdata() extracts voxel intensities as numpy
        lbl_data = lbl_obj.get_fdata()


        # -------------------------
        # Resampling
        # -------------------------
        # Original spacing

        spacing = img_obj.header.get_zooms()[:3]  # This gets the voxel size in mm e.g., (0.8, 0.8, 3.0)

        # Target spacing
        target_spacing = (1.0, 1.0, spacing[2])  # Index position for x is 0 , y is 1 and z is 2 hence keeping the spacing of z i.e., original spacing

        # =========================
        # Resampling
        # =========================
        resize_factors = (
            spacing[0] / target_spacing[0],
            spacing[1] / target_spacing[1],
            1
        ) # How the pixel reprsent the image is changing

        img_data = ndimage.zoom(img_data, resize_factors, order=1)  # linear
        lbl_data = ndimage.zoom(lbl_data, resize_factors, order=0)  # nearest

        depth = img_data.shape[-1]

        # =========================
        # Slice-wise processing
        # =========================
        for z in range(depth): # loops thru the z
            img_slice = img_data[:, :, z] # extracts one slice in one loop 
            lbl_slice = lbl_data[:, :, z]

            # Padding 
            img_slice = pad_or_crop_512(img_slice) # calling pad function
            lbl_slice = pad_or_crop_512(lbl_slice)

           

            slice_name = f"{base_name}_slice_{z:05d}.npy"

            # Save each slice as anumpy arra
            np.save(os.path.join(out_img_dir, slice_name), img_slice.astype(np.float32))
            np.save(os.path.join(out_lbl_dir, slice_name), lbl_slice.astype(np.uint8))

            total_slices += 1

    return total_slices

# =========================
# Run
# =========================
print("\nProcessing TRAIN...")
train_slices = process(train_pairs, OUT_IMAGES_TRAIN, OUT_LABELS_TRAIN)

print("\nProcessing VAL...")
val_slices = process(val_pairs, OUT_IMAGES_VAL, OUT_LABELS_VAL)

print("\nDone!")
print(f"Train slices: {train_slices}")
print(f"Val slices: {val_slices}")