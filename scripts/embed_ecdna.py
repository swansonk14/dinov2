"""Script to embed ecDNA images using DINOv2."""
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def embed_ecdna(
        data_dir: Path,
        hub_dir: str,
        model_type: str = 'dinov2_vits14',
        patch_size: int = 14,
        batch_size: int = 10,
        num_cells_per_row: int = 10,
        num_cells_per_col: int = 10,
        image_max: int = 2**16 - 1
) -> None:
    """Embed ecDNA images using DINOv2.

    :param data_dir: A directory containing the ecDNA images as .tif images.
    :param hub_dir: A directory where the DINOv2 model will be saved.
    :param model_type: The type of DINOv2 model to use.
    :param patch_size: The size of the patches used by the DINOv2 model.
    :param batch_size: The batch size.
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
    model = model.eval().cuda()

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
        cells = []

        for row in range(num_cells_per_col):  # iterate over rows
            for col in range(num_cells_per_row):  # iterate over columns
                # Get cell
                cell = image[
                       :,
                       row * cell_height: (row + 1) * cell_height,
                       col * cell_width: (col + 1) * cell_width
                ]

                # Keep cell if non-empty
                if cell.abs().sum() > 0:
                    cells.append(cell)

        # Convert cells to tensor
        cells = torch.stack(cells)  # (num_cells, 1, cell_height, cell_width)

        # Expand to three channels
        cells = cells.expand(-1, 3, -1, -1)  # (num_cells, 3, cell_height, cell_width)

        # Move to cuda
        cells = cells.cuda()

        # Run model in batches
        cell_features = []

        with torch.no_grad():
            for i in range(0, len(cells), batch_size):
                # Get batch of cells
                batch_cells = cells[i: i + batch_size]

                # Compute features
                batch_features = model(batch_cells)

                # Add to list
                cell_features.append(batch_features.cpu())

        # Concatenate features
        cell_features = torch.cat(cell_features, dim=0)  # (num_cells, feature_dim)

        # Save features
        save_path = image_path.with_suffix('.pt')
        torch.save(cell_features, save_path)


if __name__ == '__main__':
    from tap import tapify

    tapify(embed_ecdna)
