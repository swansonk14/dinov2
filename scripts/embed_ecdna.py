"""Script to embed ecDNA images using DINOv2."""
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# TODO: check the image width/height and why not a multiple of 10
# TODO: check if images are always 10x10 cells
def embed_ecdna(
        data_dir: Path,
        hub_dir: str,
        model_type: str = 'dinov2_vits14_lc',
        patch_size: int = 14,
        num_cells_per_row: int = 10,
        num_cells_per_col: int = 10,
        image_max: int = 2**16 - 1
) -> None:
    """Embed ecDNA images using DINOv2.

    :param data_dir: A directory containing the ecDNA images as .tif images.
    :param hub_dir: A directory where the DINOv2 model will be saved.
    :param model_type: The type of DINOv2 model to use.
    :param patch_size: The size of the patches used by the DINOv2 model.
    :param num_cells_per_row: The number of cells per row.
    :param num_cells_per_col: The number of cells per column.
    :param image_max: The maximum value of the image. Used to normalize the image.
    """
    # Define a transform to convert the image to a tensor and normalize it
    to_tensor = transforms.ToTensor()

    # Load model
    torch.hub.set_dir(hub_dir)
    model = torch.hub.load('facebookresearch/dinov2', model_type)

    # Move to cuda
    model = model.cuda()

    # TODO: try merging ecDNA stain and hoechst stain into one image

    # Get image paths
    image_paths = list(data_dir.glob('**/*.tif'))

    # Process each image
    for image_path in tqdm(image_paths):
        # Load the image
        image = Image.open(image_path)

        # Get image dimensions
        image_width, image_height = image.size

        # Upsample the image to the next multiple of patch_size * num_cells_per_row/col
        image_width_needed_multiple = patch_size * num_cells_per_row
        image_height_needed_multiple = patch_size * num_cells_per_col
        image = image.resize((
            image_width + (image_width_needed_multiple - image_width % image_width_needed_multiple),
            image_height + (image_height_needed_multiple - image_height % image_height_needed_multiple)
        ))

        # Get new image dimensions
        image_width, image_height = image.size

        # Convert image data type
        image = image.convert('I')

        # Convert to FloatTensor
        image = to_tensor(image)  # (1, height, width)

        # Check image values
        assert image.dtype == torch.int32
        assert image.min() >= 0
        assert image.max() <= image_max

        # Normalize to [0, 1] range
        image = image / image_max

        # Compute per cell dimensions
        cell_width = image_width // num_cells_per_row
        cell_height = image_height // num_cells_per_col

        # Split image into cells
        cells = torch.zeros((num_cells_per_row * num_cells_per_col, 1, cell_height, cell_width), dtype=torch.float)  # (num_cells, 1, cell_height, cell_width)

        for i in range(num_cells_per_row):
            for j in range(num_cells_per_col):
                cells[i * num_cells_per_row + j] = image[
                    :,
                    j * cell_height: (j + 1) * cell_height,
                    i * cell_width: (i + 1) * cell_width
                ]

        # Expand to three channels
        cells = cells.expand(-1, 3, -1, -1)  # (num_cells, 3, cell_height, cell_width)

        # Move to cuda
        cells = cells.cuda()

        # Run model
        cell_features = model(cells)  # (num_cells, hidden_dim)

        # Save features
        save_path = image_path.with_suffix('.pt')
        torch.save(cell_features, save_path)


if __name__ == '__main__':
    from tap import tapify

    tapify(embed_ecdna)
