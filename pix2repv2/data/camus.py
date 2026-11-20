"""CAMUS (2D echocardiography) datasets for Pix2Rep-v2.

Dataset layout (as distributed, or after `scripts/preprocessing/CAMUS/CAMUS_preprocess.py`):

    $CAMUS_DATASET_FOLDER/
        database_nifti/patientXXXX/
            patientXXXX_{2CH,4CH}_{ED,ES}.nii[.gz]                  (W, H)     2D image
            patientXXXX_{2CH,4CH}_{ED,ES}_gt.nii[.gz]               (W, H)     2D label map
            patientXXXX_{2CH,4CH}_half_sequence.nii[.gz]            (W, H, T)  ED -> ES frames
            patientXXXX_{2CH,4CH}_half_sequence_gt.nii[.gz]         (W, H, T)
            Info_{2CH,4CH}.cfg
        database_split/subgroup_{training,validation,testing}.txt  (400 / 50 / 50 patients)

Labels: 0 = background, 1 = LV endocardium (cavity), 2 = LV myocardium, 3 = left atrium.

Main differences with the cine-MRI datasets (ACDC, M&Ms, M&Ms-2):
    - Images are 2D (no slice dimension): ED/ES images are 2D NIfTIs, sequences are (W, H, T).
    - Native pixel spacing is very fine (0.308 mm, isotropic) and image sizes vary a lot
      (from ~320x290 to ~1180x970 pixels). All images are therefore resampled to a common
      spacing (`cfg.data.camus.target_spacing`, 0.6 mm by default), *including during
      pretraining*, so that 128x128 patches cover a similar anatomical extent at all stages.
    - Everything outside the ultrasound sector (cone) is exactly zero. Patches are sampled
      with their center inside the sector to avoid feeding empty patches to the network.
    - Ultrasound-specific photometric augmentation (multiplicative speckle-like noise) is added
      and intensity inversion is reduced/disabled (inverted B-mode images never occur in practice).
    - CAMUS ships an official train/val/test split, so no random slice-level validation split is used.
"""

import math
import os
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from einops import rearrange
from loguru import logger
from scipy.ndimage import binary_fill_holes
from torch.utils.data import Dataset

from pix2repv2.utils import utils
from pix2repv2.utils.augmentations import (
    RandomInvert,
    RandomRescaleIntensity,
    RandomSpeckleNoise,
)

CAMUS_VIEWS = ("2CH", "4CH")
CAMUS_INSTANTS = ("ED", "ES")
CAMUS_SPLITS = ("training", "validation", "testing")
CAMUS_QUALITIES = ("Good", "Medium", "Poor")

PATCH_SIZE = 128
# Images are stored in [0, 255]; anything above this threshold is considered inside the sector.
SECTOR_THRESHOLD = 1.0


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _camus_cfg(cfg) -> dict:
    """Returns the CAMUS-specific config block (cfg.data.camus) with defaults filled in."""
    defaults = {
        "target_spacing": 0.6,
        "views": list(CAMUS_VIEWS),
        "use_half_sequence_gt": False,
        "speckle_noise_std": 0.15,
        "speckle_noise_p": 0.3,
        "pretraining_random_invert_p": 0.1,
        "finetuning_random_invert_p": 0.0,
    }
    user_cfg = cfg.data.get("camus", None) if "data" in cfg else None
    if user_cfg is not None:
        defaults.update({k: v for k, v in user_cfg.items()})
    return defaults


def _resolve_nifti(path_without_ext: str) -> str:
    """Returns `<path>.nii` (preprocessed, uncompressed, mmap-able) if it exists, otherwise
    `<path>.nii.gz` (raw CAMUS release)."""
    for ext in (".nii", ".nii.gz"):
        if os.path.exists(path_without_ext + ext):
            return path_without_ext + ext
    raise FileNotFoundError(f"Neither {path_without_ext}.nii nor .nii.gz exists")


def _patient_dir(data_folder_path: str, pid: str) -> str:
    return os.path.join(data_folder_path, "database_nifti", pid)


def read_camus_split(data_folder_path: str, split: str) -> list[str]:
    """Reads the official CAMUS split file and returns the list of patient ids."""
    assert split in CAMUS_SPLITS, f"split must be one of {CAMUS_SPLITS}, got {split}"
    split_file = os.path.join(
        data_folder_path, "database_split", f"subgroup_{split}.txt"
    )
    with open(split_file) as f:
        return sorted(line.strip() for line in f if line.strip())


def parse_camus_info_cfg(info_path: str) -> dict[str, str]:
    infos = {}
    with open(info_path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            infos[key.strip()] = value.strip()
    return infos


def _spacing_2d(nii: nib.Nifti1Image) -> tuple[float, float, float]:
    """In-plane spacing (sw, sh) + a dummy depth spacing of 1.0 to match the (C, W, H, D)
    convention used by `utils.resample` (the 3rd zoom of sequences is time, not space)."""
    zooms = nii.header.get_zooms()
    return (float(zooms[0]), float(zooms[1]), 1.0)


def _load_frame(path: str, t: int | None) -> np.ndarray:
    """Lazily loads a single 2D frame (W, H) from a 2D image or a (W, H, T) sequence."""
    nii = nib.load(path, mmap=True)
    if t is None:
        frame = np.asarray(nii.dataobj)
        if frame.ndim == 3:  # some writers keep a singleton 3rd dim
            frame = frame[..., 0]
    else:
        frame = np.asarray(nii.dataobj[:, :, t])
    return frame.astype(np.float32)


def sector_mask(frame: torch.Tensor) -> torch.Tensor:
    """Binary mask of the ultrasound sector, from a (C, W, H, D) image in [0, 255].
    Pixels outside the cone are exactly zero; dark pixels inside the cone (e.g. blood pool)
    are recovered by hole filling."""
    mask = frame[0, :, :, 0].numpy() > SECTOR_THRESHOLD
    mask = binary_fill_holes(mask)
    return torch.from_numpy(mask).to(torch.int32)[None, :, :, None]


def _pad_to_min_size(subject: tio.Subject, min_size: int) -> tio.Subject:
    """Zero-pads the (W, H) dims so that a (min_size, min_size) patch can always be extracted."""
    w, h, _ = subject.spatial_shape
    pad_w, pad_h = max(0, min_size - w), max(0, min_size - h)
    if pad_w == 0 and pad_h == 0:
        return subject
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, 0, 0)
    return tio.Pad(padding, padding_mode=0)(subject)


def _build_photometric_transform(cfg_transform, camus_cfg: dict, invert_p: float):
    """Same photometric pipeline as for cine-MRI, plus multiplicative speckle-like noise.
    The bias field is kept: it is a reasonable proxy for depth-dependent attenuation /
    time-gain-compensation inhomogeneities in B-mode images."""
    return tio.Compose(
        [
            tio.RandomBiasField(
                coefficients=cfg_transform.random_field_coef,
                p=cfg_transform.random_field_p,
            ),
            # out_min_max=(Imin, Imax) to reduce the gap between rescaling at the
            # patch-level (pretraining/finetuning) and rescaling at the image-level (inference/eval)
            RandomRescaleIntensity(
                Imin_range=(0.0, 0.3),
                Imax_range=(0.7, 1.0),
                percentiles=(1, 99),
            ),
            RandomSpeckleNoise(
                std=(0.0, camus_cfg["speckle_noise_std"]),
                p=camus_cfg["speckle_noise_p"],
            ),
            tio.RandomBlur(
                std=cfg_transform.random_blur_std,
                p=cfg_transform.random_blur_p,
            ),
            tio.RandomGamma(
                log_gamma=cfg_transform.log_gamma,
                p=cfg_transform.random_gamma_p,
            ),
            tio.RandomNoise(
                std=cfg_transform.random_noise_std,
                p=cfg_transform.random_noise_p,
            ),
            RandomInvert(p=invert_p),
        ]
    )


# --------------------------------------------------------------------------------------
# Pretraining
# --------------------------------------------------------------------------------------
class CAMUSPatchSSL(Dataset):
    """Patch-based SSL pretraining dataset for CAMUS.

    Every frame of the 2CH/4CH half sequences (ED -> ES) of the patients in `split` is used
    (~15.4k frames for the official training split). Labels are *not* used.
    Each frame is resampled to `target_spacing`, then a random square patch whose center lies
    inside the ultrasound sector is extracted (side = U(min, max) * min(W, H)), resized to
    (128, 128, 1), and two photometrically augmented views are generated.
    The grid to be applied for the geometric transform is also returned.
    """

    def __init__(
        self,
        cfg: dict,
        data_folder_path: str = os.environ.get("CAMUS_DATASET_FOLDER"),
        min_patch_size_factor: float | None = None,
        max_patch_size_factor: float | None = None,
        apply_augmentations: bool = True,
        apply_elastic_transform: bool = True,
        split: str = "training",
        target_spacing: float | None = None,
        views: list[str] | None = None,
    ):
        assert data_folder_path is not None, "Set CAMUS_DATASET_FOLDER"
        self.cfg = cfg
        self.camus_cfg = _camus_cfg(cfg)
        self.data_folder_path = data_folder_path
        self.min_patch_size_factor = (
            min_patch_size_factor
            if min_patch_size_factor is not None
            else self.cfg.pretraining.min_patch_size_factor
        )
        self.max_patch_size_factor = (
            max_patch_size_factor
            if max_patch_size_factor is not None
            else self.cfg.pretraining.max_patch_size_factor
        )
        self.apply_augmentations = apply_augmentations
        self.apply_elastic_transform = apply_elastic_transform
        self.split = split
        self.target_spacing = (
            target_spacing
            if target_spacing is not None
            else float(self.camus_cfg["target_spacing"])
        )
        self.views = list(views if views is not None else self.camus_cfg["views"])

        # Precompute all (sequence path, frame index) pairs
        self.slice_indices: list[tuple[str, int]] = list()
        for pid in read_camus_split(data_folder_path, split):
            p_dir = _patient_dir(data_folder_path, pid)
            for view in self.views:
                seq_path = _resolve_nifti(
                    os.path.join(p_dir, f"{pid}_{view}_half_sequence")
                )
                T = nib.load(seq_path).shape[-1]
                self.slice_indices.extend([(seq_path, t) for t in range(T)])
        logger.info(f"CAMUS SSL ({split}): {len(self.slice_indices)} frames")

        self.resizer = tio.Resize(
            target_shape=(PATCH_SIZE, PATCH_SIZE, 1),  # shape: (W, H, D)
            image_interpolation="bspline",
        )
        self.intensity_rescaler = RandomRescaleIntensity(
            Imin_range=(0.0, 0.3),
            Imax_range=(0.7, 1.0),
            percentiles=(1, 99),
        )
        self.pretraining_transform = _build_photometric_transform(
            self.cfg.pretraining.pretraining_transform,
            self.camus_cfg,
            invert_p=self.camus_cfg["pretraining_random_invert_p"],
        )

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, t = self.slice_indices[idx]
        frame = _load_frame(path, t)  # (W, H)
        spacing_in = _spacing_2d(nib.load(path, mmap=True))

        frame_tensor = torch.tensor(
            rearrange(frame, "W H -> 1 W H 1"), dtype=torch.float32
        )
        frame_tensor, _ = utils.resample(
            img=frame_tensor,
            spacing_in=spacing_in,
            spacing_out=(self.target_spacing, self.target_spacing, 1.0),
            mode="bilinear",  # no overshoot outside the sector
        )  # shape: (C=1, W, H, D=1)

        subject = tio.Subject(
            slice_2d=tio.ScalarImage(tensor=frame_tensor),
            sampling_map=tio.LabelMap(tensor=sector_mask(frame_tensor)),
        )

        w, h, _ = subject.spatial_shape
        patch_size_factor = np.random.uniform(
            self.min_patch_size_factor,
            self.max_patch_size_factor,
        )
        side = max(16, round(min(h, w) * patch_size_factor))
        patch_size = (side, side, 1)

        # Patch center sampled inside the US sector. Fall back to uniform sampling if the
        # sector mask is empty / too close to the borders for the requested patch size.
        try:
            sampler = tio.data.WeightedSampler(
                patch_size=patch_size, probability_map="sampling_map"
            )
            patch = next(sampler._generate_patches(subject, num_patches=1))
        except Exception:
            sampler = tio.data.UniformSampler(patch_size=patch_size)
            patch = next(sampler._generate_patches(subject, num_patches=1))

        resized_patch = self.resizer(patch)  # shape: (C=1, W=128, H=128, D=1)
        patch_data = resized_patch["slice_2d"].data

        # SSL photometric augmentations
        if self.apply_augmentations:
            view1 = self.pretraining_transform(patch_data)
            view2 = self.pretraining_transform(patch_data)
        else:
            view1 = self.intensity_rescaler(patch_data)
            view2 = self.intensity_rescaler(patch_data)
        assert view1.shape == view2.shape  # shape: (C=1, W=128, H=128, D=1)

        # Rigid geometric transforms for images and feature maps (equivariance)
        affine_matrix = utils.generate_single_affine_spatial_transform(
            is_rotated=self.cfg.pretraining.is_rotated,
            is_cropped=self.cfg.pretraining.is_cropped,
            is_flipped=self.cfg.pretraining.is_flipped,
            is_translation=self.cfg.pretraining.is_translation,
            max_angle=self.cfg.pretraining.max_angle,
            max_crop=self.cfg.pretraining.max_crop,
        )
        if self.apply_elastic_transform:
            with torch.no_grad():
                grid = utils.generate_random_affine_elastic_grid(
                    p=self.cfg.pretraining.pretraining_transform.elastic_transform_p,
                    H=view1.shape[2],  # recall that view1 has shape (C, W, H, D)
                    W=view1.shape[1],
                    affine=affine_matrix,
                )  # shape: (1, W, H, 2)
        else:
            grid = F.affine_grid(
                affine_matrix.unsqueeze(0),  # add 'batch' dim ('B=1')
                rearrange(view1, "C W H D -> D C W H").shape,
                align_corners=True,
            )

        return view1, view2, grid


# --------------------------------------------------------------------------------------
# Finetuning
# --------------------------------------------------------------------------------------
class CAMUSPatchSupervised(Dataset):
    """Patch-based supervised CAMUS dataset.

    By default, uses the manually annotated ED and ES images of both views (4 images per
    patient), i.e. the standard CAMUS benchmark setting. With `use_half_sequence_gt=True`,
    every annotated frame of the half sequences (ED -> ES) is used instead.
    Images are resampled to ~`target_spacing` (randomly scaled by +/-10% as augmentation),
    then a 128x128 patch is extracted with a LabelSampler (40% background, 60% structures).
    """

    def __init__(
        self,
        cfg: dict,
        data_folder_path: str = os.environ.get("CAMUS_DATASET_FOLDER"),
        is_resampled: bool = True,
        target_spacing: float | None = None,
        apply_augmentations: bool = True,
        apply_elastic_transform: bool = True,
        # Patients for finetuning in different data regimes. If None, the whole split is used.
        selected_patients: list[str] | None = None,
        split: str = "training",  # "training" or "validation"
        views: list[str] | None = None,
        use_half_sequence_gt: bool | None = None,
    ):
        assert data_folder_path is not None, "Set CAMUS_DATASET_FOLDER"
        self.cfg = cfg
        self.camus_cfg = _camus_cfg(cfg)
        self.data_folder_path = data_folder_path
        self.is_resampled = is_resampled
        self.target_spacing = (
            target_spacing
            if target_spacing is not None
            else float(self.camus_cfg["target_spacing"])
        )
        self.apply_augmentations = apply_augmentations
        self.apply_elastic_transform = apply_elastic_transform
        self.selected_patients = selected_patients
        self.split = split
        self.views = list(views if views is not None else self.camus_cfg["views"])
        self.use_half_sequence_gt = (
            use_half_sequence_gt
            if use_half_sequence_gt is not None
            else bool(self.camus_cfg["use_half_sequence_gt"])
        )

        patient_ids = read_camus_split(data_folder_path, split)
        if self.selected_patients is not None:
            selected_set = set(self.selected_patients)
            patient_ids = [pid for pid in patient_ids if pid in selected_set]

        # (img_path, gt_path, frame index or None for 2D files, pid)
        self.slice_indices: list[tuple[str, str, int | None, str]] = list()
        for pid in patient_ids:
            p_dir = _patient_dir(data_folder_path, pid)
            for view in self.views:
                if self.use_half_sequence_gt:
                    base = os.path.join(p_dir, f"{pid}_{view}_half_sequence")
                    img_path = _resolve_nifti(base)
                    gt_path = _resolve_nifti(base + "_gt")
                    T = nib.load(img_path).shape[-1]
                    self.slice_indices.extend(
                        [(img_path, gt_path, t, pid) for t in range(T)]
                    )
                else:
                    for instant in CAMUS_INSTANTS:
                        base = os.path.join(p_dir, f"{pid}_{view}_{instant}")
                        try:
                            img_path = _resolve_nifti(base)
                            gt_path = _resolve_nifti(base + "_gt")
                        except FileNotFoundError:
                            logger.warning(
                                f"Missing {view}_{instant} image/GT for {pid}. Skipping..."
                            )
                            continue
                        self.slice_indices.append((img_path, gt_path, None, pid))
        logger.info(
            f"CAMUS supervised ({split}): {len(patient_ids)} patients, "
            f"{len(self.slice_indices)} labeled images"
        )

        self.intensity_rescaler = RandomRescaleIntensity(
            Imin_range=(0.0, 0.3),
            Imax_range=(0.7, 1.0),
            percentiles=(1, 99),
        )
        self.supervised_transform = _build_photometric_transform(
            self.cfg.finetuning.supervised_transform,
            self.camus_cfg,
            invert_p=self.camus_cfg["finetuning_random_invert_p"],
        )

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        img_path, gt_path, t, pid = self.slice_indices[idx]
        img = _load_frame(img_path, t)
        gt = _load_frame(gt_path, t)
        orig_spacing = _spacing_2d(nib.load(img_path, mmap=True))

        img_tensor = torch.tensor(rearrange(img, "W H -> 1 W H 1"), dtype=torch.float32)
        gt_tensor = torch.tensor(
            rearrange(np.rint(gt), "W H -> 1 W H 1"), dtype=torch.int32
        )
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_tensor),
            label=tio.LabelMap(tensor=gt_tensor),
        )

        actual_spacing_out = orig_spacing
        if self.is_resampled:
            # RandomScale of +/-10% around the target spacing to augment scale variability
            if self.apply_augmentations:
                s_out = np.random.uniform(
                    0.9 * self.target_spacing, 1.1 * self.target_spacing
                )
            else:
                s_out = self.target_spacing
            target_spacing = (s_out, s_out, 1.0)

            img_resampled, actual_spacing_out = utils.resample(
                img=subject["image"].data,
                spacing_in=orig_spacing,
                spacing_out=target_spacing,
                mode="bicubic",
            )  # shape: (C, W, H, D)
            # grid_sample() does not support int: cast to float and back
            label_resampled, _ = utils.resample(
                img=subject["label"].data.float(),
                spacing_in=orig_spacing,
                spacing_out=target_spacing,
                mode="nearest",
            )
            subject["image"].data = img_resampled
            subject["label"].data = label_resampled.int()

        subject = _pad_to_min_size(subject, PATCH_SIZE)

        sampler = tio.data.LabelSampler(
            patch_size=(PATCH_SIZE, PATCH_SIZE, 1),
            label_name="label",
            label_probabilities={
                0: 0.4,  # BACKGROUND
                1: 0.2,  # LV cavity
                2: 0.2,  # LV myocardium
                3: 0.2,  # Left atrium
            },
        )
        patch = next(sampler._generate_patches(subject, num_patches=1))

        if self.apply_augmentations:
            if self.apply_elastic_transform:
                affine_matrix = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
                with torch.no_grad():
                    grid = utils.generate_random_affine_elastic_grid(
                        p=self.cfg.finetuning.supervised_transform.elastic_transform_p,
                        H=patch["image"].data.shape[2],
                        W=patch["image"].data.shape[1],
                        affine=affine_matrix,
                    )  # shape: (1, W, H, 2)

                # Apply elastic transform, then photometric augmentations
                patch_to_elastic = rearrange(patch["image"].data, "C W H D -> D C W H")
                patch_to_elastic = F.grid_sample(
                    patch_to_elastic, grid, mode="bicubic", align_corners=True
                )
                patch_to_photometric = rearrange(patch_to_elastic, "D C W H -> C W H D")
                img_data = self.supervised_transform(patch_to_photometric)

                gt_to_elastic = rearrange(patch["label"].data, "C W H D -> D C W H")
                gt_to_elastic = F.grid_sample(
                    gt_to_elastic.float(), grid, mode="nearest", align_corners=True
                ).int()
                gt_data = rearrange(gt_to_elastic, "D C W H -> C W H D")
            else:
                img_data = self.supervised_transform(patch["image"].data)
                gt_data = patch["label"].data
        else:
            img_data = self.intensity_rescaler(patch["image"].data)
            gt_data = patch["label"].data

        assert img_data.shape == gt_data.shape

        return img_data, gt_data, pid, tuple(float(s) for s in actual_spacing_out)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
class CAMUSSupervisedEval(Dataset):
    """Full-image evaluation dataset for CAMUS.

    Each item is one annotated 2D image (patient x view x instant), returned with a singleton
    depth dim so that it plugs into the same (C, W, H, D) evaluation pipeline as the cine-MRI
    datasets. The image is resampled to `target_spacing` for inference; the ground truth is kept
    in its original space (predictions are resampled back before computing the Dice score).
    """

    def __init__(
        self,
        cfg: dict,
        data_folder_path: str = os.environ.get("CAMUS_DATASET_FOLDER"),
        is_resampled: bool = True,
        target_spacing: float | None = None,
        split: str = "testing",
        selected_patients: list[str] | None = None,
        views: list[str] | None = None,
    ):
        assert data_folder_path is not None, "Set CAMUS_DATASET_FOLDER"
        self.cfg = cfg
        self.camus_cfg = _camus_cfg(cfg)
        self.data_folder_path = data_folder_path
        self.is_resampled = is_resampled
        self.target_spacing = (
            target_spacing
            if target_spacing is not None
            else float(self.camus_cfg["target_spacing"])
        )
        self.split = split
        self.selected_patients = selected_patients
        self.views = list(views if views is not None else self.camus_cfg["views"])

        patient_ids = read_camus_split(data_folder_path, split)
        if selected_patients is not None:
            selected_set = set(selected_patients)
            patient_ids = [pid for pid in patient_ids if pid in selected_set]

        # (img_path, gt_path, item_id)
        self.slice_indices: list[tuple[str, str, str]] = list()
        for pid in patient_ids:
            p_dir = _patient_dir(data_folder_path, pid)
            for view in self.views:
                for instant in CAMUS_INSTANTS:
                    base = os.path.join(p_dir, f"{pid}_{view}_{instant}")
                    try:
                        img_path = _resolve_nifti(base)
                        gt_path = _resolve_nifti(base + "_gt")
                    except FileNotFoundError:
                        logger.warning(f"Missing {view}_{instant} for {pid}. Skipping...")
                        continue
                    self.slice_indices.append(
                        (img_path, gt_path, f"{pid}_{view}_{instant}")
                    )

        self.intensity_rescaler = tio.RescaleIntensity(
            out_min_max=(0, 1),
            percentiles=(1, 99),
        )

    def __len__(self):
        # 50 test patients x 2 views x 2 instants = 200 images
        return len(self.slice_indices)

    def __getitem__(self, idx):
        img_path, gt_path, item_id = self.slice_indices[idx]
        img = _load_frame(img_path, None)
        gt = _load_frame(gt_path, None)
        orig_spacing = _spacing_2d(nib.load(img_path, mmap=True))

        img_tensor = torch.tensor(rearrange(img, "W H -> 1 W H 1"), dtype=torch.float32)
        gt_tensor = torch.tensor(
            rearrange(np.rint(gt), "W H -> 1 W H 1"), dtype=torch.int32
        )

        actual_spacing_out = orig_spacing
        if self.is_resampled:
            img_tensor, actual_spacing_out = utils.resample(
                img=img_tensor,
                spacing_in=orig_spacing,
                spacing_out=(self.target_spacing, self.target_spacing, 1.0),
                mode="bicubic",
            )

        subject = tio.Subject(image=tio.ScalarImage(tensor=img_tensor))
        img_data = self.intensity_rescaler(subject)["image"].data

        return (
            img_data,  # (C=1, W', H', D=1), resampled
            gt_tensor,  # (C=1, W, H, D=1), original space
            item_id,
            orig_spacing,
            tuple(float(s) for s in actual_spacing_out),
        )


# --------------------------------------------------------------------------------------
# Patient selection for low-data regimes
# --------------------------------------------------------------------------------------
def _patient_quality(p_dir: str) -> str:
    """Worst image quality across the 2CH and 4CH views of a patient."""
    qualities = []
    for view in CAMUS_VIEWS:
        info_path = os.path.join(p_dir, f"Info_{view}.cfg")
        if os.path.exists(info_path):
            qualities.append(parse_camus_info_cfg(info_path).get("ImageQuality"))
    qualities = [q for q in qualities if q in CAMUS_QUALITIES]
    if not qualities:
        return "Medium"
    return max(qualities, key=CAMUS_QUALITIES.index)


def select_nested_patient_ids(
    cfg: dict,
    data_folder_path: str = os.environ.get("CAMUS_DATASET_FOLDER"),
) -> list[str]:
    """Returns the CAMUS training patients to use for finetuning.

    `cfg.finetuning.num_patients` is interpreted as a *percentage* of the 400 official
    training patients (as for M&Ms / M&Ms-2), e.g. 1 -> 4 patients, 100 -> all 400.

    Properties:
        - Stratified by image quality (Good / Medium / Poor, worst of both views), the CAMUS
          equivalent of the pathology stratification used for ACDC
        - Nested across data regimes (1% ⊂ 2% ⊂ 5% ⊂ ...)
        - Reproducible given cfg.data.random_seed
    """
    patient_ids = read_camus_split(data_folder_path, "training")

    groups: dict[str, list[str]] = {q: [] for q in CAMUS_QUALITIES}
    for pid in patient_ids:
        groups[_patient_quality(_patient_dir(data_folder_path, pid))].append(pid)

    rng = random.Random(cfg.data.random_seed)
    # Global ordering: within each group, patients are shuffled and assigned evenly spaced
    # ranks in (0, 1). Sorting by rank interleaves the groups proportionally to their size,
    # so any prefix of the ordering is (approximately) stratified, and prefixes are nested.
    ranked: list[tuple[float, float, str]] = []
    for quality in CAMUS_QUALITIES:
        ids = groups[quality]
        rng.shuffle(ids)
        for i, pid in enumerate(ids):
            ranked.append(((i + 0.5) / len(ids), rng.random(), pid))
    ranked.sort()

    pct = float(cfg.finetuning.num_patients)
    assert 0 < pct <= 100, "Percentage of finetuning patients must be in (0, 100]"
    num_patients = max(1, math.ceil(pct / 100.0 * len(patient_ids)))

    return [pid for _, _, pid in ranked[:num_patients]]
