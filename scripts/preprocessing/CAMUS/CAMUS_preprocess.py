"""Preprocess the CAMUS public release for fast on-the-fly loading.

The raw release stores every image as gzip-compressed float32 NIfTI. Reading a single frame of a
half sequence then requires decompressing the whole sequence (~35-65 ms per frame), which makes
patch-based pretraining I/O-bound. This script converts every file to an *uncompressed* uint8
NIfTI (.nii) that nibabel can memory-map (< 1 ms per frame), while keeping the native resolution,
affine and spacing (resampling is done on-the-fly by the datasets, cf. cfg.data.camus).
Info_*.cfg files and the official split are copied as-is.

Usage:
    python scripts/preprocessing/CAMUS/CAMUS_preprocess.py \
        --input_dir data/CAMUS_public --output_dir data/CAMUS_processed
    export CAMUS_DATASET_FOLDER=data/CAMUS_processed

The datasets in pix2repv2/data/camus.py accept both layouts (.nii or .nii.gz), so this step is
optional but strongly recommended for pretraining.
"""

import argparse
import glob
import os
import shutil
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from loguru import logger
from tqdm import tqdm


def convert_file(args: tuple[str, str]) -> str | None:
    src, dst = args
    if os.path.exists(dst):
        return None
    nii = nib.load(src)
    data = np.asarray(nii.dataobj)

    rounded = np.rint(data)
    if np.allclose(data, rounded) and data.min() >= 0 and data.max() <= 255:
        data, dtype = rounded.astype(np.uint8), np.uint8
    else:  # should not happen for CAMUS, but never lose information
        data, dtype = data.astype(np.float32), np.float32
        logger.warning(f"{src}: non-uint8 values, keeping float32")

    header = nii.header.copy()
    header.set_data_dtype(dtype)
    header.set_slope_inter(1.0, 0.0)
    out = nib.Nifti1Image(data, nii.affine, header)
    out.header.set_zooms(nii.header.get_zooms())

    tmp = dst + ".tmp.nii"
    nib.save(out, tmp)
    os.replace(tmp, dst)  # atomic: no half-written files if interrupted
    return dst


def main(input_dir: str, output_dir: str, num_workers: int):
    src_nifti_dir = os.path.join(input_dir, "database_nifti")
    dst_nifti_dir = os.path.join(output_dir, "database_nifti")
    assert os.path.isdir(src_nifti_dir), f"{src_nifti_dir} not found"

    # Official split + license files
    shutil.copytree(
        os.path.join(input_dir, "database_split"),
        os.path.join(output_dir, "database_split"),
        dirs_exist_ok=True,
    )

    jobs = []
    for p_dir in sorted(glob.glob(os.path.join(src_nifti_dir, "patient*"))):
        pid = os.path.basename(p_dir)
        out_p_dir = os.path.join(dst_nifti_dir, pid)
        os.makedirs(out_p_dir, exist_ok=True)

        for f in glob.glob(os.path.join(p_dir, "*.cfg")) + glob.glob(
            os.path.join(p_dir, "*.md")
        ):
            shutil.copy2(f, out_p_dir)

        for src in sorted(glob.glob(os.path.join(p_dir, "*.nii.gz"))):
            dst = os.path.join(out_p_dir, os.path.basename(src)[: -len(".gz")])
            jobs.append((src, dst))

    logger.info(f"Converting {len(jobs)} NIfTI files with {num_workers} workers")
    with Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(convert_file, jobs, chunksize=8), total=len(jobs)):
            pass
    logger.info(f"Done. export CAMUS_DATASET_FOLDER={os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/CAMUS_public")
    parser.add_argument("--output_dir", type=str, default="data/CAMUS_processed")
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.num_workers)
