import os
import glob
import torch
import time
from datetime import datetime
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandRotate90d,
    RandFlipd,
    EnsureTyped,
    NormalizeIntensityd,   # Z- score normalization
)


# =========================
# CONFIGURATION
# =========================
DATA_DIR = r"Data\2d_slice_5{3}"

TRAIN_IMAGES = os.path.join(DATA_DIR, "imagesTr")
TRAIN_LABELS = os.path.join(DATA_DIR, "labelsTr")

VAL_IMAGES = os.path.join(DATA_DIR, "imagesVal")
VAL_LABELS = os.path.join(DATA_DIR, "labelsVal")

NUM_CLASSES = 4
BATCH_SIZE = 8
NUM_WORKERS = 0
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# START TIME
# =========================
start_time = time.time()
print("Training started at:", datetime.now())

# =========================
# LOAD FILE PATHS
# =========================
train_images = sorted(glob.glob(os.path.join(TRAIN_IMAGES, "*.npy")))
train_labels = sorted(glob.glob(os.path.join(TRAIN_LABELS, "*.npy")))

val_images = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.npy")))
val_labels = sorted(glob.glob(os.path.join(VAL_LABELS, "*.npy")))

train_files = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]
val_files = [{"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)]

print(f"Training on {len(train_files)} slices | Validating on {len(val_files)} slices")

# =========================
# TRANSFORMS
# =========================
train_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader="NumpyReader"),
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"], nonzero= False, channel_wise=True),  # z score normalization
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),

    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader="NumpyReader"),
    EnsureChannelFirstd(keys=["image", "label"]),

    NormalizeIntensityd(keys=["image"], nonzero= False, channel_wise=True),

    EnsureTyped(keys=["image", "label"]),
])

# =========================
# DATASETS & LOADERS
# =========================
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =========================
# MODEL
# =========================
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=NUM_CLASSES,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(DEVICE)

# =========================
# LOSS, OPTIMIZER, METRIC
# =========================

#  Dice + Cross Entropy combined
loss_function = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    lambda_dice=0.5,
    lambda_ce=0.5
)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
dice_metric = DiceMetric(include_background=True, reduction="mean")

post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
post_label = AsDiscrete(to_onehot=NUM_CLASSES)
# =========================
# TRAINING LOOP
# =========================
best_metric = -1

for epoch in range(MAX_EPOCHS):
    print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")

    epoch_start = time.time()

    # -------- TRAIN --------
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1

        inputs = batch_data["image"].to(DEVICE)
        labels = batch_data["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, labels)  
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Average Train Loss: {epoch_loss/step:.4f}")

    # -------- VALIDATION --------
    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(DEVICE)
            val_labels = val_data["label"].to(DEVICE)

            val_outputs = model(val_inputs)

            # ✅ Proper formatting
            # 🔥 CORRECT MONAI WAY
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels  = [post_label(i) for i in decollate_batch(val_labels)]

            dice_metric(y_pred=val_outputs, y=val_labels)
    metric = dice_metric.aggregate().item()
    print(f"Validation Mean Dice: {metric:.4f}")

    if metric > best_metric:
        best_metric = metric
        torch.save(model.state_dict(), "best_model_8{a}.pth")
        print(" --> Saved new best model!")

    epoch_end = time.time()
    print(f"Epoch Time: {epoch_end - epoch_start:.2f} sec")

# =========================
# TOTAL TIME
# =========================
end_time = time.time()

print("\nTraining complete!")
print(f"Best Validation Dice: {best_metric:.4f}")

total_time = end_time - start_time

print("\nTotal Training Time:")
print(f"{total_time:.2f} seconds")
print(f"{total_time/60:.2f} minutes")
print(f"{total_time/3600:.2f} hours")

print("Training ended at:", datetime.now())