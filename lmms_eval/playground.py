import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

def plot_tensor(tensor: torch.Tensor, cmap='viridis', save_path='output/plot_tensor.svg'):
    """
    Plots a 2D PyTorch tensor as a 3D surface.
    
    Parameters:
        tensor (torch.Tensor): A 2D tensor to plot.
        cmap (str): The matplotlib colormap to use for coloring the surface.
    """
    # Ensure the tensor is on CPU and convert it to a numpy array
    if tensor.is_cuda:
        tensor = tensor.cpu()
    data = tensor.detach().numpy()
    
    # Create x and y coordinates based on tensor dimensions
    n_rows, n_cols = data.shape
    x = np.arange(n_cols)
    y = np.arange(n_rows)
    X, Y = np.meshgrid(x, y)
    
    # Create the figure and a 3D axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with face colors according to the tensor value
    surface = ax.plot_surface(X, Y, data, cmap=cmap, edgecolor='none')
    
    # Set the z-axis limits based on the tensor's min and max values
    print(np.min(data), np.max(data))
    ax.set_zlim(np.min(data), np.max(data))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surface, shrink=0.5, aspect=5)
    
    # Set labels for clarity
    ax.set_xlabel('Channel')
    ax.set_ylabel('Token')
    ax.set_zlabel('Value')
    ax.set_title(save_path)
    
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory
 

def plot_similarity(tensor: torch.Tensor, save_path='output/plot_similarity.svg'):
    """
    Plots a similarity matrix as a heatmap.
    
    Parameters:
        tensor (torch.Tensor): a n x d tensor (n samples, d features)
    """
    # Normalize the tensor
    tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = tensor @ tensor.T  # (n x d) @ (d x n) → (n x n)

    # Ensure the tensor is on CPU and convert it to numpy
    similarity_matrix = similarity_matrix.cpu().detach().numpy()

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the heatmap with a colorbar
    cax = ax.matshow(similarity_matrix, cmap='viridis')
    fig.colorbar(cax)

    # Set labels for clarity
    ax.set_xlabel('Token')
    ax.set_ylabel('Token')
    ax.set_title('Similarity Matrix')

    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory

def similarity_sparsify(tensor: torch.Tensor, threshold):
    pass

if __name__ == '__main__':
    # load .pt file
    data = torch.load('data/llava_vid_videomme_sample.pt')
    # read a line from a file
    # path = 'output/llava_vid_test_split_index.txt'
    # with open(path, 'r') as file:
    #     line = file.readline().strip()
        # matches = re.findall(r'\d+', line)

    before_length = 14
    token_length = 64 * 13 * 14

    tensor = data[0].squeeze(0)
    tensor = tensor[before_length:before_length + token_length, :]
    tensor = tensor.reshape(64, 13, 14, 3584)
    tensor = tensor.permute(1, 2, 0, 3).contiguous()
    tensor = tensor.reshape(-1, 3584)
    print(tensor.shape)


        
        # visualize the tensor
        # path = 'output/act_{}.svg'.format(key)
        # plot_tensor(value, save_path=path)
        # path = 'output/similarity_{}.svg'.format(key)
        # plot_similarity(value, save_path=path)
        
