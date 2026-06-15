import os
import SimpleITK as sitk
import numpy as np
from skimage import measure
from skimage.measure import label
from scipy.ndimage import binary_fill_holes, gaussian_filter
import plotly.graph_objects as go
import plotly.io as pio


# =========================
# 🔥 FORCE BROWSER POPUP
# =========================
pio.renderers.default = "browser"

# =========================
# 📂 PATHS
# =========================
pred_path = r"Data\Inference_7{b}\duke_534_0001.nii"
gt_path   = r"Data\Expert_segmentation\Testing_dataset\duke_534.nii.gz"
img_path  = r"Data\Testing_data\imagesTs\duke_534_0001.nii"

# =========================
# 🧠 EXTRACT NAMES
# =========================
file_name = os.path.basename(pred_path)

patient_id = file_name.replace("_0001.nii", "") \
                      .replace(".nii.gz", "") \
                      .replace(".nii", "")

inference_name = os.path.basename(os.path.dirname(pred_path))

# =========================
# LOAD DATA
# =========================
pred_np = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
gt_np   = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
img_np  = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

# =========================
# EXTRACT TUMOR
# =========================
pred_tumor = (pred_np == 1)
gt_tumor   = (gt_np == 1)

if np.sum(pred_tumor) == 0 or np.sum(gt_tumor) == 0:
    print("⚠️ Tumor not found → exiting")
    exit()

# =========================
# 🔥 SMOOTH BREAST SURFACE
# =========================
tissue = img_np > np.percentile(img_np, 60)

tissue = gaussian_filter(tissue.astype(float), sigma=2)

tissue = tissue > 0.3

labels = label(tissue)
largest_cc = np.argmax(np.bincount(labels.flat)[1:]) + 1
body = (labels == largest_cc)

body = binary_fill_holes(body)

body_smooth = gaussian_filter(body.astype(float), sigma=1)

# =========================
# CREATE SURFACES
# =========================
verts_pred, faces_pred, _, _ = measure.marching_cubes(pred_tumor, level=0.5)
verts_gt, faces_gt, _, _     = measure.marching_cubes(gt_tumor, level=0.5)
verts_body, faces_body, _, _ = measure.marching_cubes(body_smooth, level=0.5)

# =========================
# MESHES
# =========================
mesh_body = go.Mesh3d(
    x=verts_body[:, 2],
    y=verts_body[:, 1],
    z=verts_body[:, 0],
    i=faces_body[:, 0],
    j=faces_body[:, 1],
    k=faces_body[:, 2],
    color='lightpink',
    opacity=0.1,
    name='Breast Surface',
    lighting=dict(
        ambient=0.6,
        diffuse=0.8,
        roughness=0.4,
        specular=0.3
    )
)

# =========================
# GROUND TRUTH (PURPLE)
# =========================
mesh_gt = go.Mesh3d(
    x=verts_gt[:, 2],
    y=verts_gt[:, 1],
    z=verts_gt[:, 0],
    i=faces_gt[:, 0],
    j=faces_gt[:, 1],
    k=faces_gt[:, 2],
    color="#55FDFE",   # cyan
    opacity=0.6,
    name='Ground Truth'
)

# =========================
# PREDICTION (RED)
# =========================
mesh_pred = go.Mesh3d(
    x=verts_pred[:, 2],
    y=verts_pred[:, 1],
    z=verts_pred[:, 0],
    i=faces_pred[:, 0],
    j=faces_pred[:, 1],
    k=faces_pred[:, 2],
    color='#E5383B',
    opacity=0.6,
    name=inference_name
)

# =========================
# FIGURE
# =========================
fig = go.Figure(data=[mesh_body, mesh_gt, mesh_pred])

# =========================
# 📷 ISOMETRIC CAMERA
# =========================
iso_camera = dict(
    eye=dict(x=1.8, y=1.8, z=1.8),
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0)
)

# =========================
# 🎛️ OPACITY SLIDER
# =========================
steps = []

for opacity in np.linspace(0, 1, 8):
    step = dict(
        method="restyle",
        args=[{"opacity": opacity}, [0]],
        label=f"{opacity:.2f}"
    )
    steps.append(step)

sliders = [
    dict(
        active=2,
        currentvalue={"prefix": "Breast Opacity: "},
        pad={"t": 50},
        steps=steps
    )
]

# =========================
# LAYOUT
# =========================
fig.update_layout(

    title=f"{patient_id} | {inference_name}<br>3D Tumor + Breast Surface",

scene=dict(

    xaxis=dict(
        showticklabels=False,
        title="",
        showgrid=True,
        gridcolor="white",
        zeroline=False,
        showbackground=True,
        backgroundcolor="rgb(240,240,240)"
    ),

    yaxis=dict(
        showticklabels=False,
        title="",
        showgrid=True,
        gridcolor="white",
        zeroline=False,
        showbackground=True,
        backgroundcolor="rgb(240,240,240)"
    ),

    zaxis=dict(
        showticklabels=False,
        title="",
        showgrid=True,
        gridcolor="white",
        zeroline=False,
        showbackground=True,
        backgroundcolor="rgb(240,240,240)"
    ),

    aspectmode="cube",

    camera=iso_camera
),
    sliders=sliders,

    updatemenus=[

        dict(
            type="buttons",

            buttons=[

                dict(
                    label="Isometric View",

                    method="relayout",

                    args=[{
                        "scene.camera": iso_camera
                    }]
                )

            ],

            direction="left",

            x=0.25,
            y=1.15,

            showactive=False
        )

    ],

    margin=dict(
        l=0,
        r=0,
        b=0,
        t=40
    )
)



# =========================
# SINGLE BROWSER OPEN
# =========================
fig.show()