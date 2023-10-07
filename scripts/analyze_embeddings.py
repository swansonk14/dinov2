from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm
# from umap import UMAP


# TODO: remove all black images

def analyze_embeddings(
        embeddings_dir: Path = Path('../data/ecdna_images')
) -> None:
    # Load embeddings
    plate_to_condition_to_stain_to_embeddings: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for embeddings_path in tqdm(list(embeddings_dir.glob('**/*.pt'))):
        plate, condition, stain, index = embeddings_path.stem.split('_')
        plate_to_condition_to_stain_to_embeddings[plate][condition][stain].append(torch.load(embeddings_path))

    # Get lists of plates, conditions, stains, and embeddings
    plates, conditions, stains, all_embeddings = [], [], [], []
    for plate, condition_to_stain_to_embeddings in plate_to_condition_to_stain_to_embeddings.items():
        for condition, stain_to_embeddings in condition_to_stain_to_embeddings.items():
            for stain, embeddings_list in stain_to_embeddings.items():
                for embeddings in embeddings_list:
                    num_cells = embeddings.shape[0]
                    plates += [plate] * num_cells
                    conditions += [condition] * num_cells
                    stains += [stain] * num_cells
                    all_embeddings.append(embeddings)

    # Convert to Pandas and Torch
    cell_data = pd.DataFrame({
        'plate': plates,
        'condition': conditions,
        'stain': stains
    })
    embeddings = torch.cat(all_embeddings)

    # Limit number of cells
    # TODO: remove this
    # max_cells = 2000
    # cell_data = cell_data.iloc[:max_cells].copy()
    # embeddings = embeddings[:max_cells]

    # Limit stain
    # TODO: remove this
    # stain = 'DNAFISH'
    # stain_mask = cell_data['stain'] == 'DNAFISH'
    # cell_data = cell_data[stain_mask].copy()
    # embeddings = embeddings[stain_mask]

    # t-SNE/UMAP embed the embeddings
    # umap_embeddings = UMAP().fit_transform(embeddings)
    tsne_embeddings = TSNE().fit_transform(embeddings)

    # Add embeddings to cell data
    cell_data['tsne_0'] = tsne_embeddings[:, 0]
    cell_data['tsne_1'] = tsne_embeddings[:, 1]

    # Visualize the embeddings
    for key in ['plate', 'condition', 'stain']:
        sns.scatterplot(data=cell_data, x='tsne_0', y='tsne_1', hue=key)
        plt.title(f't-SNE Cell Embeddings by {key.title()}')
        plt.show()


if __name__ == '__main__':
    from tap import tapify

    tapify(analyze_embeddings)
