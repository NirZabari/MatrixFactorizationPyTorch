import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from dataset import MovieLensDataSet

def init_embedding_vectors(dataloader, latent_dim):
    users_embedding = {userId: torch.rand(size=(1, latent_dim)).float().requires_grad_(True)
                       for userId in dataloader.dataset.get_unique_users()}
    items_embedding = {itemId: torch.rand(size=(1, latent_dim)).float().requires_grad_(True)
                       for itemId in dataloader.dataset.get_unique_items()}
    return users_embedding, items_embedding

def get_embedding_from_dict(d, keys):
    return [d[k] for k in keys]

def update_embeddings_to_dict(d, keys, new_values):
    for i, k in enumerate(keys):
        d[k] = new_values[i].detach()

def get_dataloader(batch_size):
    dataset = MovieLensDataSet(movie_lens_dir=os.path.join(os.path.dirname(__file__), "movielens-100k"))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def plot_loss(loss_epochs, title=""):
    num_epochs = len(loss_epochs)
    for label in ['train_loss','test_loss']:
        plt.plot(np.arange(num_epochs), [v[label] for v in loss_epochs.values()], label=label)

    plt.title(title)
    plt.legend()
    plt.show()