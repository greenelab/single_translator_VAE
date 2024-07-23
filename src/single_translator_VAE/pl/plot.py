from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def loss_plots(
    loss_to_plot: Optional[str] = "summed",
    training_losses: np.ndarray = None,
    validation_losses: np.ndarray = None,
    all_train_losses: Optional[dict] = None,
    all_val_losses: Optional[dict] = None,
):
    """

    Plot the training loss over epochs.

    Parameters
    ----------
    - training_losses (list of floats): A list containing the loss recorded at each epoch.
    - title (str): Title of the plot.

    This function creates a line plot representing the training loss across epochs,
    helping to visualize the model's learning progress over time.

    """
    if loss_to_plot == "summed":
        epochs = range(1, len(training_losses) + 1)
        plt.figure(figsize=(14, 8))
        plt.plot(epochs, training_losses, label="Training Loss", marker="o", color="b", markersize=1)
        plt.plot(epochs, validation_losses, label="Validation Loss", marker="o", color="r", markersize=1)
        plt.title("Loss (total) Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    else:
        epochs = range(1, len(training_losses) + 1)
        if loss_to_plot == "classification":
            train_ind_loss = all_train_losses["class_loss"]
            val_ind_loss = all_val_losses["class_loss"]
        if loss_to_plot == "kl":
            train_ind_loss = all_train_losses["kl_local"]
            val_ind_loss = all_val_losses["kl_local"]
        if loss_to_plot == "reconstruction":
            train_ind_loss = all_train_losses["reconstruction_loss"]
            val_ind_loss = all_val_losses["reconstruction_loss"]

        plt.figure(figsize=(14, 8))
        plt.plot(epochs, train_ind_loss, label="Training Loss", marker="o", color="b", markersize=1)
        plt.plot(epochs, val_ind_loss, label="Validation Loss", marker="o", color="r", markersize=1)
        plt.title(f"Training Loss: {loss_to_plot.capitalize()}, Over Epochs", fontweight="bold", fontsize=13)
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.grid(True)
        plt.legend()
        plt.show()


def results_PCA(
    model,
    adata,
    input_tensor,
    labels_tensor,
    batch_tensor: torch.Tensor,
    labels_key: str = "labels_key",
):
    """
    Plot the resulting PCA with reconstructions.

    Parameters
    ----------
    model
        VAE model to use for process.
    input_tensor
        Tensor to encode and decode for reconstruciton and plot for input data.
    adata
        AnnData used in input_tensor to get categories to plot.
    labels_tensor
        Tensor for labels to be used for encoding and decoding.
    batch_tensor
        Tensor for batches to be used for encoding and decoding.
    labels_key
        Colunm name string in adata.obs to get labels from.


    Returns
    -------
    original_data
        Original data plotted as numpy array.
    latent_space
        Laent space numpy array.
    reconstructed_data
        Original data after encoding and decoding with labels and numpy array.

    """
    # Ensure the labels_key column is of type 'category'
    if labels_key in adata.obs:
        adata.obs[labels_key] = adata.obs[labels_key].astype("category")
    else:
        # Handle the case where 'labels_key' is not present or mislabeled
        AssertionError("labels_key used not found in adata.obs")

        # model in evaluation model:
    model.eval()
    # Perform inference to get latent space, library, and reconstruction
    with torch.no_grad():
        outputs = model.module.inference(input_tensor, labels_tensor, batch_index=batch_tensor)
        reconstructed_data = model.module.generative(
            outputs["z"], outputs["library"], labels_tensor, batch_index=batch_tensor
        )["px_rate"]

    # Convert to numpy for PCA and plotting
    latent_space_ = outputs["z"].numpy()
    reconstructed_data_ = reconstructed_data.numpy()
    original_data_ = adata.X.toarray()

    scaler = StandardScaler()
    original_data = scaler.fit_transform(original_data_)
    reconstructed_data = scaler.transform(reconstructed_data_)
    scaler = StandardScaler()
    latent_space = scaler.fit_transform(latent_space_)

    # Apply PCA
    pca = PCA(n_components=2)
    pca_input = pca.fit_transform(original_data)
    pca_reconstructed = pca.transform(reconstructed_data)
    pca = PCA(n_components=2)
    pca_latent = pca.fit_transform(latent_space)

    # Set up colors based on labels
    color_map = {dtype: plt.cm.tab10(i) for i, dtype in enumerate(adata.obs[labels_key].cat.categories)}
    colors = [color_map[label] for label in adata.obs[labels_key]]

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].scatter(pca_input[:, 0], pca_input[:, 1], c=colors, alpha=0.5)
    axes[0].set_title("PCA of Input Data", fontweight="bold", fontsize=12)
    axes[1].scatter(pca_latent[:, 0], pca_latent[:, 1], c=colors, alpha=0.5)
    axes[1].set_title("PCA of Latent Space", fontweight="bold", fontsize=12)
    axes[2].scatter(pca_reconstructed[:, 0], pca_reconstructed[:, 1], c=colors, alpha=0.5)
    axes[2].set_title("PCA of Reconstructed Data", fontweight="bold", fontsize=12)

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    # Create a legend
    handles = [Patch(color=color, label=label) for label, color in color_map.items()]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

    return original_data_, latent_space_, reconstructed_data_


def pca_of_augmentation(adata_train, adata_test, labels_key="origin"):
    """
    Plot PCA for augmented and original data.

    Parameters
    ----------
    adata_train : AnnData
        AnnData object containing the training data.
    adata_test : AnnData
        AnnData object containing the testing data.
    labels_key : str
        Column in .obs to color the plots by.
    """
    adata_test.obs[labels_key] = adata_test.obs[labels_key].astype("category")
    adata_train.obs[labels_key] = adata_train.obs[labels_key].astype("category")

    # Standardize and PCA transform the training data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(adata_train.X.toarray())
    pca = PCA(n_components=2)
    pca_train = pca.fit_transform(train_scaled)

    # Standardize and PCA transform the testing data
    test_scaled = scaler.fit_transform(adata_test.X.toarray())
    pca_test = pca.transform(test_scaled)

    # Plotting
    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot PCA for test data colored by data_type
    test_colors = [plt.cm.tab10(i) for i in adata_test.obs["data_type"].cat.codes]
    axes[0].scatter(pca_test[:, 0], pca_test[:, 1], c=test_colors, alpha=0.5)
    axes[0].set_title("PCA of Test Data (by data_type)")

    # Plot PCA for train data colored by data_type
    train_colors = [plt.cm.tab10(i) for i in adata_train.obs["data_type"].cat.codes]
    axes[1].scatter(pca_train[:, 0], pca_train[:, 1], c=train_colors, alpha=0.5)
    axes[1].set_title("PCA of Train Data (by data_type)")

    # Plot PCA for train data colored by origin
    origin_colors = [plt.cm.tab10(i) for i in adata_train.obs[labels_key].cat.codes]
    axes[2].scatter(pca_train[:, 0], pca_train[:, 1], c=origin_colors, alpha=0.5)
    axes[2].set_title("PCA of Train Data (by origin)")

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    # Create legends
    handles_test = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=plt.cm.tab10(i), markersize=10)
        for i in range(len(adata_test.obs["data_type"].cat.categories))
    ]
    labels_test = adata_test.obs["data_type"].cat.categories
    axes[0].legend(handles_test, labels_test, title="data_type", loc="best")

    handles_train_data_type = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=plt.cm.tab10(i), markersize=10)
        for i in range(len(adata_train.obs["data_type"].cat.categories))
    ]
    labels_train_data_type = adata_train.obs["data_type"].cat.categories
    axes[1].legend(handles_train_data_type, labels_train_data_type, title="data_type", loc="best")

    handles_train_origin = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=plt.cm.tab10(i), markersize=10)
        for i in range(len(adata_train.obs[labels_key].cat.categories))
    ]
    labels_train_origin = adata_train.obs[labels_key].cat.categories
    axes[2].legend(handles_train_origin, labels_train_origin, title="origin", loc="best")

    plt.tight_layout()
    plt.show()


def reconstruction_PCA(
    model,
    adata,
    input_tensor,
    labels_tensor,
):
    """

    Plot the reconstruction PCA.

    Parameters
    ----------
    model
        VAE model to use for process.
    input_tensor
        Tensor to encode and decode for reconstruciton and plot for input data.
    adata
        AnnData used in input_tensor to get categories to plot.
    labels_tensor
        Tensor for labels to be used for encoding and decoding.
    labels_key
        Colunm name string in adata.obs to get labels from.

    Returns
    -------
    original_data
        Original data plotted as numpy array.
    latent_space
        Laent space numpy array.
    reconstructed_data
        Original data after encoding and decoding with labels and numpy array.

    """
    # model in evaluation model:
    model.eval()
    # Perform inference to get latent space, library, and reconstruction
    with torch.no_grad():
        outputs = model.module.inference(input_tensor, labels_tensor.unsqueeze(1))
        latent_space = outputs["z"]
        library = outputs["library"]
        reconstructed_data = model.module.generative(latent_space, library, labels_tensor.unsqueeze(1))["px_rate"]

    # Convert to numpy for PCA and plotting
    reconstructed_data_ = reconstructed_data.cpu().numpy()
    original_data_ = adata.X.toarray()

    scaler = StandardScaler()
    original_data = scaler.fit_transform(original_data_)
    reconstructed_data = scaler.transform(reconstructed_data_)

    # Apply PCA
    pca = PCA(n_components=2)
    pca_input = pca.fit_transform(original_data)
    pca_reconstructed = pca.transform(reconstructed_data)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_input[:, 0], pca_input[:, 1], c="b", alpha=0.5, label="Original")
    plt.scatter(pca_reconstructed[:, 0], pca_reconstructed[:, 1], c="r", alpha=0.5, label="Reconstructed")
    plt.title("PCA of Input Data With Reconstruction", fontweight="bold", fontsize=12)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()

    return original_data_, latent_space, reconstructed_data_


def rmse_per_celltype(rmse_dict):
    """
    Plots violin plots of RMSE for each cell type based on multiple RMSE values per cell type.

    Parameters
    ----------
    - rmse_dict: Dictionary with cell types as keys and a list of RMSE values for each cell of that type.
    """
    # Prepare data for plotting
    data = [(cell_type, rmse) for cell_type, rmses in rmse_dict.items() for rmse in rmses]
    cell_types, rmses = zip(*data)

    # Create a violin plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(x=cell_types, y=rmses)
    plt.xticks(rotation=45)
    plt.xlabel("Cell Type", fontweight="bold")
    plt.ylabel("RMSE", fontweight="bold")
    plt.title("RMSE (Input vs. Reconstructed Data) per Cell Type", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.show()


def transformed_pca_data(original_sn, original_sc, sn_to_sn, sc_to_sc, sc_to_sn, sn_to_sc):
    """

    Plots PCA of the input and reconstructed datasets using the same PCA fitting.

    Parameters
    ----------
    original_sn : numpy array
        Original single nucleus data.
    original_sc : numpy array
        Original single cell data.
    sn_to_sn : numpy array
        Reconstructed single nucleus to single nucleus data.
    sc_to_sc : numpy array
        Reconstructed single cell to single cell data.
    sc_to_sn : numpy array
        Transformed single cell to single nucleus data.
    sn_to_sc : numpy array
        Transformed single nucleus to single cell data.

    """
    # Combine original datasets for scaling and PCA
    original_data = np.concatenate((original_sn, original_sc))
    reconstructed_data = np.concatenate((sn_to_sn, sc_to_sc, sc_to_sn, sn_to_sc))

    # Scale all data based on the scale of the original data
    scaler = StandardScaler()
    original_data_scaled = scaler.fit_transform(original_data)
    reconstructed_data_scaled = scaler.transform(reconstructed_data)

    # Apply PCA using fit_transform on the combined original data, then transform reconstructed data
    pca = PCA(n_components=2)
    pca.fit(original_data_scaled)  # Fit PCA on the original data

    pca_original_sn = pca.transform(original_data_scaled[: len(original_sn)])
    pca_original_sc = pca.transform(original_data_scaled[len(original_sn) :])

    # Ensure the reconstruction uses the same PCA object
    pca_sn_to_sn = pca.transform(reconstructed_data_scaled[: len(sn_to_sn)])
    pca_sc_to_sc = pca.transform(reconstructed_data_scaled[len(sn_to_sn) : len(sn_to_sn) + len(sc_to_sc)])
    pca_sc_to_sn = pca.transform(
        reconstructed_data_scaled[len(sn_to_sn) + len(sc_to_sc) : len(sn_to_sn) + len(sc_to_sc) + len(sc_to_sn)]
    )
    pca_sn_to_sc = pca.transform(reconstructed_data_scaled[-len(sn_to_sc) :])

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First plot for original data
    axes[0].scatter(pca_original_sn[:, 0], pca_original_sn[:, 1], c="blue", label="Original SN", marker="o")
    axes[0].scatter(pca_original_sc[:, 0], pca_original_sc[:, 1], c="red", label="Original SC", marker="o")
    axes[0].set_title("PCA of Original Data", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # Second plot for reconstructed and transformed data
    axes[1].scatter(pca_sn_to_sn[:, 0], pca_sn_to_sn[:, 1], c="blue", label="SN to SN: Recon.", marker="o")
    axes[1].scatter(pca_sc_to_sc[:, 0], pca_sc_to_sc[:, 1], c="red", label="SC to SC: Recon.", marker="o")
    axes[1].scatter(pca_sc_to_sn[:, 0], pca_sc_to_sn[:, 1], c="lightblue", label="SC to SN: Transf.", marker="*")
    axes[1].scatter(pca_sn_to_sc[:, 0], pca_sn_to_sc[:, 1], c="pink", label="SN to SC: Transf.", marker="*")
    axes[1].set_title("PCA of Reconstructed and Transformed Data", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()


def transformed_pca_data_with_input(original_sn, original_sc, sn_to_sn, sc_to_sc, sc_to_sn, sn_to_sc):
    """
    Plots PCA of the input and reconstructed datasets using the same PCA fitting.

    Parameters
    ----------
    original_sn : numpy array
        Original single nucleus data.
    original_sc : numpy array
        Original single cell data.
    sn_to_sn : numpy array
        Reconstructed single nucleus to single nucleus data.
    sc_to_sc : numpy array
        Reconstructed single cell to single cell data.
    sc_to_sn : numpy array
        Transformed single cell to single nucleus data.
    sn_to_sc : numpy array
        Transformed single nucleus to single cell data.

    """
    # Combine original datasets for scaling and PCA
    original_data = np.concatenate((original_sn, original_sc))
    reconstructed_data = np.concatenate((sn_to_sn, sc_to_sc, sc_to_sn, sn_to_sc))

    # Scale all data based on the scale of the original data
    scaler = StandardScaler()
    original_data_scaled = scaler.fit_transform(original_data)
    reconstructed_data_scaled = scaler.transform(reconstructed_data)

    # Apply PCA using fit_transform on the combined original data, then transform reconstructed data
    pca = PCA(n_components=2)
    pca.fit(original_data_scaled)  # Fit PCA on the original data

    pca_original_sn = pca.transform(original_data_scaled[: len(original_sn)])
    pca_original_sc = pca.transform(original_data_scaled[len(original_sn) :])

    # Ensure the reconstruction uses the same PCA object
    pca_sn_to_sn = pca.transform(reconstructed_data_scaled[: len(sn_to_sn)])
    pca_sc_to_sc = pca.transform(reconstructed_data_scaled[len(sn_to_sn) : len(sn_to_sn) + len(sc_to_sc)])
    pca_sc_to_sn = pca.transform(
        reconstructed_data_scaled[len(sn_to_sn) + len(sc_to_sc) : len(sn_to_sn) + len(sc_to_sc) + len(sc_to_sn)]
    )
    pca_sn_to_sc = pca.transform(reconstructed_data_scaled[-len(sn_to_sc) :])

    # Plotting
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    # First plot for original data
    axes[0].scatter(pca_original_sn[:, 0], pca_original_sn[:, 1], c="blue", label="Original SN", marker="o", alpha=0.3)
    axes[0].scatter(pca_original_sc[:, 0], pca_original_sc[:, 1], c="red", label="Original SC", marker="o", alpha=0.3)
    # Second plot for reconstructed and transformed data
    axes[0].scatter(pca_sn_to_sn[:, 0], pca_sn_to_sn[:, 1], c="blue", label="SN to SN: Recon.", marker="*", alpha=0.3)
    axes[0].scatter(pca_sc_to_sc[:, 0], pca_sc_to_sc[:, 1], c="red", label="SC to SC: Recon.", marker="*", alpha=0.3)
    axes[0].scatter(
        pca_sc_to_sn[:, 0], pca_sc_to_sn[:, 1], c="lightblue", label="SC to SN: Transf.", marker="*", alpha=0.3
    )
    axes[0].scatter(pca_sn_to_sc[:, 0], pca_sn_to_sc[:, 1], c="pink", label="SN to SC: Transf.", marker="*", alpha=0.3)
    axes[0].set_title("PCA of Input Data, Reconstructed and Transformed", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()
