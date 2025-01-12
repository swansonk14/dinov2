"""Compute t-SNE/UMAP on cell embeddings and visualize them."""
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm
from umap import UMAP


def analyze_embeddings(
    embeddings_dir: Path,
    save_dir: Path,
    max_cells: Optional[int] = None,
    selected_stain: Optional[str] = None,
    embedding_type: Literal["tsne", "umap"] = "tsne",
) -> None:
    """Compute t-SNE/UMAP on cell embeddings and visualize them.

    :param embeddings_dir: Path to directory containing cell embeddings.
    :param save_dir: Path to directory where plots will be saved.
    :param max_cells: The maximum number of cells to analyze.
    :param selected_stain: The stain to analyze. If None, all stains will be analyzed.
    :param embedding_type: The type of embedding to use. Either 'tsne' or 'umap'.
    """
    # Load embeddings
    well_to_condition_to_stain_to_embeddings: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for embeddings_path in tqdm(list(embeddings_dir.glob("**/*.pt"))):
        well, condition, stain, index = embeddings_path.stem.split("_")
        well_to_condition_to_stain_to_embeddings[well][condition][stain].append(torch.load(embeddings_path))

    # Get lists of wells, conditions, stains, and embeddings
    wells, conditions, stains, all_embeddings = [], [], [], []
    for well, condition_to_stain_to_embeddings in well_to_condition_to_stain_to_embeddings.items():
        for condition, stain_to_embeddings in condition_to_stain_to_embeddings.items():
            for stain, embeddings_list in stain_to_embeddings.items():
                for embeddings in embeddings_list:
                    num_cells = embeddings.shape[0]
                    wells += [well] * num_cells
                    conditions += [condition] * num_cells
                    stains += [stain] * num_cells
                    all_embeddings.append(embeddings)

    # Convert to Pandas and Torch
    cell_data = pd.DataFrame({"well": wells, "condition": conditions, "stain": stains})
    embeddings = torch.cat(all_embeddings)

    # Optionally, limit number of cells
    if max_cells is not None:
        cell_data = cell_data.iloc[:max_cells].copy()
        embeddings = embeddings[:max_cells]

    # Optionally, limit stain
    if selected_stain is not None:
        stain_mask = cell_data["stain"] == selected_stain
        cell_data = cell_data[stain_mask].copy()
        embeddings = embeddings[stain_mask]

    # t-SNE/UMAP embed the embeddings
    if embedding_type == "tsne":
        reduced_embeddings = TSNE().fit_transform(embeddings)
    elif embedding_type == "umap":
        reduced_embeddings = UMAP().fit_transform(embeddings)
    else:
        raise ValueError(f"Invalid embedding_type: {embedding_type}")

    # Add embeddings to cell data
    cell_data["embed_0"] = reduced_embeddings[:, 0]
    cell_data["embed_1"] = reduced_embeddings[:, 1]

    # Visualize the embeddings and save
    save_dir.mkdir(parents=True, exist_ok=True)

    for key in tqdm(["well", "condition", "stain"], desc="Plotting well/condition/stain embeddings"):
        sns.scatterplot(data=cell_data, x="embed_0", y="embed_1", hue=key, s=5)
        plt.title(f"{embedding_type.upper()} Cell Embeddings by {key.title()}")
        plt.savefig(save_dir / f"{embedding_type}_{key}.pdf", bbox_inches="tight")
        plt.close()

    blue = sns.color_palette()[0]
    red = sns.color_palette()[3]
    gray = sns.color_palette()[7]

    for well in tqdm(sorted(cell_data["well"].unique()), desc="Plotting well embeddings"):
        well_data = cell_data[cell_data["well"] == well]
        well_control_data = well_data[well_data["condition"] == "GFP"]
        well_treatment_data = well_data[well_data["condition"] == "mCherry"]
        remaining_data = cell_data[cell_data["well"] != well]

        sns.scatterplot(data=remaining_data, x="embed_0", y="embed_1", color=gray, s=5, label="Other")
        sns.scatterplot(data=well_control_data, x="embed_0", y="embed_1", color=blue, s=20, label=f"{well} Control")
        sns.scatterplot(data=well_treatment_data, x="embed_0", y="embed_1", color=red, s=20, label=f"{well} Treatment")

        plt.title(f"{embedding_type.upper()} Cell Embeddings for Well {well}")
        plt.savefig(save_dir / f"{embedding_type}_well_{well}.pdf", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    from tap import tapify

    tapify(analyze_embeddings)
