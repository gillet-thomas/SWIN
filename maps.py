import os
import yaml
import torch
import datetime
import seaborn as sns
from tqdm import tqdm

import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from SWIN import Model, ADNISwiFTDataset

# New function for t-SNE plotting
def plot_embeddings(model, dataloader, config, mode):
    all_embeddings = []
    all_labels = []
    method = config['map_method']

    print(f"Collecting embeddings for {mode}...")
    with torch.no_grad():
        for i, (fmri_sequence, target) in enumerate(tqdm(dataloader)):
            fmri_sequence = fmri_sequence.float().to(device=f"cuda:{cuda_id}")
            embeddings = model.get_embeddings(fmri_sequence) # t-sne expects flattened embeddings (2D)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Collected {len(all_embeddings)} embeddings of shape {all_embeddings.shape[1]}")

    print(f"Performing {method} dimensionality reduction...")
    if method == 'tsne':
        perplexity = 30 # 30 or 50
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    elif method == 'umap':
        neighbors = 100 # 50 or 100
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=neighbors, min_dist=0.1)

    embedding_results = reducer.fit_transform(all_embeddings)

    # Plotting
    plt.figure(figsize=(10, 8))
    
    label_names = {0: 'Category 0', 1: 'Category 1'}
    if config['map_task'] == 'sex':
        label_names = {0: 'Female', 1: 'Male'}
    elif config['map_task'] == 'age_group':
        label_names = {0: 'Young', 1: 'Old'}

    df_tsne = pd.DataFrame(embedding_results, columns=[f'{method} Dimension 1', f'{method} Dimension 2'])
    df_tsne['Label_Name'] = [label_names[int(label)] for label in all_labels]

    sns.scatterplot(
        x=f'{method} Dimension 1', y=f'{method} Dimension 2',
        hue="Label_Name",
        palette=sns.color_palette("hls", len(label_names)),
        data=df_tsne,
        legend="full",
        alpha=0.8
    )
    plt.title(f'{method.upper()} of {config["map_task"]} Embeddings')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    
    # Save the plot
    embedding_plot_dir = './visualization/maps'
    os.makedirs(embedding_plot_dir, exist_ok=True)
    # embedding_plot_path = os.path.join(embedding_plot_dir, f'{method}_{config["map_task"]}_{mode}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png')
    embedding_plot_path = os.path.join(embedding_plot_dir, f'{method}_{config["map_task"]}_{mode}_{perplexity}.png')
    # embedding_plot_path = os.path.join(embedding_plot_dir, f'{method}_{config["map_task"]}_{mode}_{neighbors}.png')
    plt.savefig(embedding_plot_path, dpi=300)
    print(f"{method.upper()} plot saved to {embedding_plot_path}")
    

# Load model
cuda_id = 2
config = yaml.safe_load(open("data/config.yaml", "r"))
best_model_path = config[f'best_model_{config["map_task"]}']
model = Model(config).to(device=f"cuda:{cuda_id}")
model.load_state_dict(torch.load(best_model_path))
model.eval()
print(f"Using model from {best_model_path}")

# Run t-SNE on validation, training and test datasets
data_val = ADNISwiFTDataset(config, mode='val')
val_loader = torch.utils.data.DataLoader(data_val, batch_size=config['eval_batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, prefetch_factor=2)
plot_embeddings(model, val_loader, config, "val")

data_train = ADNISwiFTDataset(config, mode='train')
train_loader = torch.utils.data.DataLoader(data_train, batch_size=config['eval_batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, prefetch_factor=2)
plot_embeddings(model, train_loader, config, "train")

data_test = ADNISwiFTDataset(config, mode='test')
test_loader = torch.utils.data.DataLoader(data_test, batch_size=config['eval_batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, prefetch_factor=2)
plot_embeddings(model, test_loader, config, "test")