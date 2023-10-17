"""Merge single-channel stain images into a single image (where the third channel is black)."""
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_image_I_16_to_L(image: Image) -> Image:
    """Convert a PIL Image in I;16 mode to L mode.

    :param image: A PIL Image in I;16 mode.
    :return: A PIL Image in L mode.
    """
    # Convert Image to NumPy array
    image_array = np.array(image)

    # Convert to L mode with bitshift
    image_array = (image_array >> 8).astype(np.uint8)

    # Convert NumPy array to Image
    image = Image.fromarray(image_array)

    return image


def merge_stains_single(dnafish_image_path: Path, save_dir: Path) -> None:
    """Merge single-channel stain images into a single image (where the third channel is black).

    :param dnafish_image_path: Path to a DNAFISH image.
    :param save_dir: Path to a directory where the merged images will be saved.
    """
    # Get Hoechst image path
    well, condition, stain, index = dnafish_image_path.stem.split("_")
    hoechst_image_path = dnafish_image_path.parent / f"{well}_{condition}_hoechst_{index}.tif"

    # Load images
    dnafish_image = Image.open(dnafish_image_path)
    hoechst_image = Image.open(hoechst_image_path)

    # Create black image for third channel
    black_image = Image.new(dnafish_image.mode, dnafish_image.size, color=0)

    # Convert formats
    dnafish_image = convert_image_I_16_to_L(dnafish_image)
    hoechst_image = convert_image_I_16_to_L(hoechst_image)
    black_image = convert_image_I_16_to_L(black_image)

    # Merge images
    merged_image = Image.merge("RGB", (black_image, dnafish_image, hoechst_image,),)

    # Save merged image
    save_path = save_dir / dnafish_image_path.parent.name / f"{well}_{condition}_merged_{index}.tif"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    merged_image.save(save_path)
    merged_image.close()


def merge_stains(data_dir: Path, save_dir: Path) -> None:
    """Merge single-channel stain images into a single image (where the third channel is black).

    :param data_dir: Path to a directory containing directories with single-channel stain images.
    :param save_dir: Path to a directory where the merged images will be saved.
    """
    # Get image paths for DNAFISH stains
    dnafish_image_paths = []
    for image_path in tqdm(list(data_dir.glob("**/*.tif"))):
        well, condition, stain, index = image_path.stem.split("_")
        if stain == "DNAFISH":
            dnafish_image_paths.append(image_path)

    # Load, merge, and save two-stain images
    merge_stains_single_partial = partial(merge_stains_single, save_dir=save_dir)
    with Pool() as pool:
        list(tqdm(pool.imap(merge_stains_single_partial, dnafish_image_paths),
                  total=len(dnafish_image_paths), desc="Merging stains"))


if __name__ == "__main__":
    from tap import tapify

    tapify(merge_stains)
