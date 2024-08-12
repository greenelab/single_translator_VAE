from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import Patch
from openTSNE import TSNEEmbedding, affinity, initialization
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def deconvolution_tsne(normalized_pseudo_df, ref_df):
    """
    Generate t-SNE plots of reference data frames projected into the t-SNE space of normalized pseudobulks.

    This function visualizes the alignment of reference cell type expression profiles with pseudobulk data
    using t-Distributed Stochastic Neighbor Embedding (t-SNE). The function projects both the normalized
    pseudobulk data and the reference data into a shared t-SNE space and plots the results for comparison.

    Parameters
    ----------
    normalized_pseudo_df : pandas.DataFrame
        DataFrame containing the normalized pseudobulk expression data. Each row represents a sample,
        and each column represents a gene.
    ref_df : pandas.DataFrame
        DataFrame containing the reference expression data. Each row represents a gene, and each column
        represents a cell type or sample.

    Returns
    -------
    None
        The function generates and displays a t-SNE plot comparing the reference and pseudobulk data.
        The plot is displayed using matplotlib.

    Notes
    -----
    - The t-SNE embeddings are optimized separately for the pseudobulk and reference data.
    - The PerplexityBasedNN method is used to compute the affinities, and PCA is used for initial embedding.
    - The function uses a high perplexity value of 60 to capture the global structure of the data.
    - The reference data is transformed into the pseudobulk t-SNE space for visualization.
    """
    # tsne plots of reference dataframes projected in normalized pseudobulks TSNE
    # figures
    plt.figure(figsize=(12, 5))
    # data used
    x_train = normalized_pseudo_df.values
    x_test = ref_df.T.values

    # Compute the affinities between data points
    affinities_train = affinity.PerplexityBasedNN(
        x_train,
        perplexity=60,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )
    # initialize coordinates for embedd.
    init_train = initialization.pca(x_train, random_state=42)
    embedding_train = TSNEEmbedding(
        init_train,
        affinities_train,
        negative_gradient_method="fft",
        n_jobs=8,
        verbose=True,
    )
    # optimize embedding
    embedding_train = embedding_train.optimize(n_iter=500)
    # transform both in train embedd.
    tsne_train = embedding_train.transform(x_train)
    tsne_test = embedding_train.transform(x_test)
    plt.scatter(tsne_train[:, 0], tsne_train[:, 1], label="Pseudobulks tSNE")
    plt.scatter(tsne_test[:, 0], tsne_test[:, 1], label="Reference tSNE")
    plt.title("Reference in Pseudobulks TSNE")
    plt.xlabel("tSNE 1")
    plt.ylabel("tSNE 2")
    plt.legend()


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
    color_map = {dtype: plt.cm.tab20(i) for i, dtype in enumerate(adata.obs[labels_key].cat.categories)}
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


def transformed_pca_data(
    original_sn: Optional[np.ndarray],
    original_sc: Optional[np.ndarray],
    sn_to_sn: Optional[np.ndarray],
    sc_to_sc: Optional[np.ndarray],
    sc_to_sn: Optional[np.ndarray],
    sn_to_sc: Optional[np.ndarray],
):
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
    original_data = []
    if original_sn is not None:
        original_data.append(original_sn)
    if original_sc is not None:
        original_data.append(original_sc)

    if len(original_data) > 0:
        original_data = np.concatenate(original_data)
    else:
        original_data = None

    # Combine reconstructed datasets
    reconstructed_data = []
    if sn_to_sn is not None:
        reconstructed_data.append(sn_to_sn)
    if sc_to_sc is not None:
        reconstructed_data.append(sc_to_sc)
    if sc_to_sn is not None:
        reconstructed_data.append(sc_to_sn)
    if sn_to_sc is not None:
        reconstructed_data.append(sn_to_sc)

    if len(reconstructed_data) > 0:
        reconstructed_data = np.concatenate(reconstructed_data)
    else:
        reconstructed_data = None

    # Scale all data based on the scale of the original data if available
    if original_data is not None:
        scaler = StandardScaler()
        original_data_scaled = scaler.fit_transform(original_data)
        if reconstructed_data is not None:
            reconstructed_data_scaled = scaler.transform(reconstructed_data)
    else:
        original_data_scaled = None
        reconstructed_data_scaled = None

    # Apply PCA using fit_transform on the combined original data, then transform reconstructed data
    if original_data_scaled is not None:
        pca = PCA(n_components=2)
        pca.fit(original_data_scaled)  # Fit PCA on the original data

        pca_original_sn = pca.transform(original_data_scaled[: len(original_sn)]) if original_sn is not None else None
        pca_original_sc = pca.transform(original_data_scaled[len(original_sn) :]) if original_sc is not None else None

        # Ensure the reconstruction uses the same PCA object
        offset = 0
        pca_sn_to_sn = pca.transform(reconstructed_data_scaled[: len(sn_to_sn)]) if sn_to_sn is not None else None
        offset += len(sn_to_sn) if sn_to_sn is not None else 0
        pca_sc_to_sc = (
            pca.transform(reconstructed_data_scaled[offset : offset + len(sc_to_sc)]) if sc_to_sc is not None else None
        )
        offset += len(sc_to_sc) if sc_to_sc is not None else 0
        pca_sc_to_sn = (
            pca.transform(reconstructed_data_scaled[offset : offset + len(sc_to_sn)]) if sc_to_sn is not None else None
        )
        offset += len(sc_to_sn) if sc_to_sn is not None else 0
        pca_sn_to_sc = pca.transform(reconstructed_data_scaled[offset:]) if sn_to_sc is not None else None
    else:
        pca_original_sn = None
        pca_original_sc = None
        pca_sn_to_sn = None
        pca_sc_to_sc = None
        pca_sc_to_sn = None
        pca_sn_to_sc = None

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First plot for original data
    if pca_original_sn is not None:
        axes[0].scatter(pca_original_sn[:, 0], pca_original_sn[:, 1], c="blue", label="Original SN", marker="o")
    if pca_original_sc is not None:
        axes[0].scatter(pca_original_sc[:, 0], pca_original_sc[:, 1], c="red", label="Original SC", marker="o")
    axes[0].set_title("PCA of Original Data", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # Second plot for reconstructed and transformed data
    if pca_sn_to_sn is not None:
        axes[1].scatter(pca_sn_to_sn[:, 0], pca_sn_to_sn[:, 1], c="blue", label="SN to SN: Recon.", marker="o")
    if pca_sc_to_sc is not None:
        axes[1].scatter(pca_sc_to_sc[:, 0], pca_sc_to_sc[:, 1], c="red", label="SC to SC: Recon.", marker="o")
    if pca_sc_to_sn is not None:
        axes[1].scatter(pca_sc_to_sn[:, 0], pca_sc_to_sn[:, 1], c="lightblue", label="SC to SN: Transf.", marker="*")
    if pca_sn_to_sc is not None:
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


def deconvolution_results(evaluation_results):
    """
    Plot RMSE and Pearson's correlation for each reference dataset.

    This function generates bar plots for the RMSE and Pearson's correlation coefficient for each reference dataset,
    allowing for visual comparison of the performance of each reference.

    Parameters
    ----------
    evaluation_results : dict
        A dictionary where each key corresponds to a reference type and each value is another dictionary containing
        the RMSE and Pearson's correlation coefficient for that reference.

    """
    # Extracting data for plotting
    ref_names = list(evaluation_results.keys())
    rmse_values = [evaluation_results[ref]["RMSE"] for ref in ref_names]
    correlation_values = [evaluation_results[ref]["Correlation"] for ref in ref_names]

    # Define custom colors for each reference
    colors = ["red", "blue", "pink"]
    palette = dict(zip(ref_names, colors))

    # Plot RMSE
    plt.figure(figsize=(10, 5))
    sns.barplot(x=ref_names, y=rmse_values, palette=palette, hue=ref_names, dodge=False, alpha=0.6)
    plt.title("RMSE for Each Reference")
    plt.ylabel("RMSE")
    plt.xlabel("Reference")
    plt.ylim(0, max(rmse_values) * 1.1)
    plt.legend([], [], frameon=False)  # Hides the legend
    plt.show()

    # Plot Pearson's Correlation
    plt.figure(figsize=(10, 5))
    sns.barplot(x=ref_names, y=correlation_values, palette=palette, hue=ref_names, dodge=False, alpha=0.6)
    plt.title("Pearson's Correlation for Each Reference")
    plt.ylabel("Correlation")
    plt.xlabel("Reference")
    plt.ylim(0, 1)
    plt.legend([], [], frameon=False)  # Hides the legend
    plt.show()


def estimated_vs_real_proportions(calc_prop_tot, prop_df):
    """
    Plot scatter plots of estimated versus real cell type proportions for each reference dataset.

    This function creates scatter plots comparing the estimated proportions against the real proportions for each reference.
    A red dashed diagonal line represents perfect agreement.

    Parameters
    ----------
    calc_prop_tot : dict of pd.DataFrame
        A dictionary containing the estimated proportions of cell types for each reference.
        Each key corresponds to a reference type, and each value is a DataFrame where rows are samples and columns are cell types.
    prop_df : pd.DataFrame
        DataFrame containing the known proportions of cell types for each sample. Each row corresponds to a sample, and each column corresponds to a cell type.

    """
    for ref_name, est_prop_df in calc_prop_tot.items():
        # Ensure the indices and columns match
        est_prop_df = est_prop_df.reindex(prop_df.index)
        est_prop_df = est_prop_df[prop_df.columns]

        # Scatter plot for each reference
        plt.figure(figsize=(8, 8))
        plt.scatter(prop_df.values.flatten(), est_prop_df.values.flatten(), alpha=0.5)
        plt.title(f"Estimated vs Real Proportions ({ref_name})")
        plt.xlabel("Real Proportions")
        plt.ylabel("Estimated Proportions")
        plt.plot([0, 1], [0, 1], "r--")  # Diagonal line for perfect agreement
        plt.grid(True)
        plt.show()
