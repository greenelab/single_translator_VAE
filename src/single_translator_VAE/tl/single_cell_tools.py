import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import ray as ray
import scanpy as sc
import torch
import torch.nn
from scipy.stats import poisson

import single_translator_VAE as sv

sys.path.insert(1, "../../")
sys.path.insert(1, "../")
sys.path.insert(1, "../../../../../")


def split_ID(row):
    """Takes in one row of DataFrame, returns the right cell name in metadata of MBC"""
    return row.split("-")[4]


def split_ID_2(row):
    """Takes in one row of DataFrame, returns the right cell name in metadata of MBC"""
    return row.split("-")[-1]


def merge_strings(row):
    """Takes in one row of DataFrame, returns the merged strings"""
    return row + "-1"


def prep_files_for_deconvolution(bulks, reference, path, file_name, deconvolution_method, genes, cell_types):
    """
    Prepare and save the data for deconvolution methods.

    Parameters
    ----------
    bulks (Pandas dataframe): The bulk data.
    reference (AnnData): The reference data.
    path (str): The directory where the data should be saved.
    file_name (str): The base name of the file.
    deconvolution_method (str): The deconvolution method ('bayesprism', 'nnls', 'cibersortx').

    Returns
    -------
    dict: Paths of the created data files.
    """
    genes = reference.var.index
    cell_types = reference.obs.cell_types.unique()
    reference = reference.X.toarray()

    method_path = os.path.join(path, deconvolution_method, "input_data")
    if not os.path.exists(method_path):
        os.makedirs(method_path)

    file_paths = {}

    if deconvolution_method == "cibersortx":
        reference_data = np.vstack([np.append("gene_ids", cell_types), np.column_stack([genes, reference])])
        ref_file_path = os.path.join(method_path, f"{file_name}_reference.txt")
        np.savetxt(ref_file_path, reference_data, fmt="%s", delimiter="\t")
        file_paths["reference"] = ref_file_path

        if bulks is not None:
            bulks_data = np.vstack([np.append("gene_ids", np.arange(bulks.shape[1])), np.column_stack([genes, bulks])])
            bulks_file_path = os.path.join(method_path, f"{file_name}_bulks.txt")
            np.savetxt(bulks_file_path, bulks_data, fmt="%s", delimiter="\t")
            file_paths["bulks"] = bulks_file_path

    elif deconvolution_method == "nnls":
        reference_data = np.hstack([np.array([["gene_ids"] + list(genes)]).T, np.vstack([cell_types, reference]).T])
        ref_file_path = os.path.join(method_path, f"{file_name}_reference.txt")
        np.savetxt(ref_file_path, reference_data, fmt="%s", delimiter="\t")
        file_paths["reference"] = ref_file_path

        if bulks is not None:
            bulks_data = np.hstack(
                [np.array([["gene_ids"] + list(genes)]).T, np.vstack([np.arange(bulks.shape[1]), bulks]).T]
            )
            bulks_file_path = os.path.join(method_path, f"{file_name}_bulks.txt")
            np.savetxt(bulks_file_path, bulks_data, fmt="%s", delimiter="\t")
            file_paths["bulks"] = bulks_file_path

    elif deconvolution_method == "bayesprism":
        reference_df = pd.DataFrame(reference, index=genes, columns=cell_types)
        ref_file_path = os.path.join(method_path, f"{file_name}_reference.csv")
        reference_df.to_csv(ref_file_path, index_label=False)
        file_paths["reference"] = ref_file_path

        if bulks is not None:
            bulks_df = pd.DataFrame(bulks, index=genes, columns=np.arange(bulks.shape[1]))
            bulks_file_path = os.path.join(method_path, f"{file_name}_bulks.csv")
            bulks_df.to_csv(bulks_file_path, index_label=False)
            file_paths["bulks"] = bulks_file_path

    return file_paths


def make_pseudobulks(adata, number_of_bulks, num_cells, prop_type, noise):
    """
    Create pseudobulks from single-cell RNA-seq data.

    Parameters
    ----------
    adata (anndata.AnnData): Single-cell RNA-seq data.
    number_of_bulks (int): Number of pseudobulks to create.
    num_cells (int): Number of cells to use for each pseudobulk.
    prop_type (str): Type of proportions ('random' or 'real').
    noise (bool): Whether to add Gaussian noise to the pseudobulks.

    Returns
    -------
    pandas.DataFrame: DataFrame of pseudobulks.
    pandas.DataFrame: DataFrame of cell type proportions for each pseudobulk.

    """
    cell_types = adata.obs["cell_types"].unique()
    gene_ids = adata.var.index
    pseudobulks = []
    proportions = []

    for _ in range(number_of_bulks):
        if prop_type == "random":
            prop_vector = np.random.dirichlet(np.ones(len(cell_types)))
        elif prop_type == "real":
            cell_type_counts = adata.obs["cell_types"].value_counts(normalize=True)
            prop_vector = cell_type_counts[cell_types].values
        else:
            raise ValueError("prop_type must be either 'random' or 'real'")

        cell_counts = (prop_vector * num_cells).astype(int)
        while np.any(cell_counts == 0):
            prop_vector = np.random.dirichlet(np.ones(len(cell_types)))
            cell_counts = (prop_vector * num_cells).astype(int)

        sampled_cells = []
        for cell_type, count in zip(cell_types, cell_counts):
            sampled_cells.append(
                adata[adata.obs["cell_types"] == cell_type]
                .X[
                    np.random.choice(
                        adata[adata.obs["cell_types"] == cell_type].shape[0],
                        count,
                        replace=len(adata[adata.obs["cell_types"] == cell_type]) < count,
                    ),
                    :,
                ]
                .toarray()
            )

        pseudobulk = np.sum(np.vstack(sampled_cells), axis=0)
        if noise:
            pseudobulk += np.random.normal(0, 0.05, pseudobulk.shape)
            pseudobulk = np.clip(pseudobulk, 0, None)  # Ensure non-negative values

        pseudobulks.append(pseudobulk)
        proportions.append(prop_vector)

    pseudobulks_df = pd.DataFrame(pseudobulks, columns=gene_ids)
    proportions_df = pd.DataFrame(proportions, columns=cell_types)

    return pseudobulks_df, proportions_df


def augment_data(adata, factor=3, noise_level=0.1):
    """
    Duplicates single-cell data and adds Poisson noise to the duplicated data.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the single-cell data.
    factor : int
        Number of times to duplicate the data.
    noise_level : float
        Noise level to add to the duplicated data.

    Returns
    -------
    combined_adata : AnnData
        AnnData object with original and duplicated data with added noise.
    """
    # Check if the single-cell data is already extracted
    if "single_cell" not in adata.obs["data_type"].unique():
        raise ValueError("Single-cell data not found in adata.obs['data_type']")

    # Extract single-cell data
    sc_data = adata[adata.obs["data_type"] == "single_cell"].copy()

    # Function to add Poisson noise
    def add_poisson_noise(matrix, noise_level):
        noise = poisson.rvs(matrix * noise_level, size=matrix.shape)
        return matrix + noise

    # Duplicate and add noise
    duplicated_matrices = [add_poisson_noise(sc_data.X.toarray(), noise_level) for _ in range(factor)]

    # Convert back to sparse matrices
    duplicated_adatas = [
        sc.AnnData(X=sc_data.X.__class__(matrix), obs=sc_data.obs.copy(), var=sc_data.var.copy())
        for matrix in duplicated_matrices
    ]

    # Set 'origin' for duplicated data
    for i, ad in enumerate(duplicated_adatas):
        ad.obs["origin"] = f"duplicated_{i}"

    # Add 'origin' to the original data
    sc_data.obs["origin"] = "original"

    # Combine the original and duplicated data
    combined_adata = sc.concat([sc_data] + duplicated_adatas, join="outer")

    # Recombine with the non-single-cell data if any
    non_sc_data = adata[adata.obs["data_type"] != "single_cell"].copy()
    if non_sc_data.shape[0] > 0:
        combined_adata = sc.concat([combined_adata, non_sc_data], join="outer")

    return combined_adata


def transform(adata, model, target_label, obs_to_transform="labels_key", target_batch=None):
    """
    Transforms the labels of the AnnData object and encodes-decodes the data using a VAE model.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    model : torch.nn.Module
        Trained VAE model.
    obs_to_transform : str
        The key in adata.obs to be transformed.
    target_label : str
        New label to transform the data into.

    Returns
    -------
    numpy.ndarray
        Transformed data as numpy array after encoding and decoding.
    """
    # Update the labels in adata
    adata.obs[obs_to_transform] = target_label

    if target_batch is not None:
        adata.obs["batch"] = target_label

    # Prepare data and labels as per the updated adata
    input_tensor, labels_tensor, batch_tensor = sv.pp.prepare_data_and_labels(adata, obs_to_transform)

    # Model in evaluation mode
    model.eval()

    # Perform inference to get latent space, library, and reconstruction
    with torch.no_grad():
        if batch_tensor is None:
            outputs = model.module.inference(input_tensor, labels_tensor)
            latent_space = outputs["z"]
            library = outputs["library"]
            reconstructed_data = model.module.generative(latent_space, library, labels_tensor)["px_rate"]
        else:
            outputs = model.module.inference(input_tensor, labels_tensor, batch_index=batch_tensor)
            latent_space = outputs["z"]
            library = outputs["library"]
            reconstructed_data = model.module.generative(
                latent_space, library, labels_tensor, batch_index=batch_tensor
            )["px_rate"]

    return reconstructed_data.numpy()


def calculate_rmse(original_data, reconstructed_data):
    """
    Calculates the Root Mean Square Error (RMSE) between two datasets.

    Parameters
    ----------
    - original_data: numpy array of the original data.
    - reconstructed_data: numpy array of the reconstructed data.

    Returns
    -------
    - rmse_per_gene: RMSE calculated for each gene (column).
    - rmse_per_cell: RMSE calculated for each cell (row).
    - overall_rmse: Overall RMSE across all cells and genes.
    """
    # Ensure that both arrays are numpy arrays and have the same shape
    if original_data.shape != reconstructed_data.shape:
        raise ValueError("Original and reconstructed data must have the same shape.")

    # Calculate RMSE per gene
    mse_per_gene = np.mean((original_data - reconstructed_data) ** 2, axis=0)
    rmse_per_gene = np.sqrt(mse_per_gene)

    # Calculate RMSE per cell
    mse_per_cell = np.mean((original_data - reconstructed_data) ** 2, axis=1)
    rmse_per_cell = np.sqrt(mse_per_cell)

    # Calculate overall RMSE
    mse_overall = np.mean((original_data - reconstructed_data) ** 2)
    overall_rmse = np.sqrt(mse_overall)

    return rmse_per_gene, rmse_per_cell, overall_rmse


def calculate_rmse_per_celltype(adata, original_data, reconstructed_data):
    """
    Calculates the Root Mean Square Error (RMSE) per cell for each cell type.

    Parameters
    ----------
    - adata: An AnnData object containing the cell type information in adata.obs['cell_types'].
    - original_data: numpy array of the original data.
    - reconstructed_data: numpy array of the reconstructed data.

    Returns
    -------
    - rmse_dict: Dictionary with cell types as keys and a list of RMSE values for each cell of that type.

    """
    if original_data.shape != reconstructed_data.shape:
        raise ValueError("Original and reconstructed data must have the same shape.")

    cell_types = adata.obs["cell_types"]
    unique_cell_types = np.unique(cell_types)
    rmse_dict = {}

    for cell_type in unique_cell_types:
        indices = np.where(cell_types == cell_type)[0]
        mse_per_cell = (original_data[indices] - reconstructed_data[indices]) ** 2
        rmse_per_cell = np.sqrt(np.mean(mse_per_cell, axis=1))  # RMSE per cell
        rmse_dict[cell_type] = rmse_per_cell

    return rmse_dict


# Funct to calculate RMSE
def rmse(y, y_pred):
    """
    Calculate RMSE:

    y : Real values
    y_pred: Predicted Values

    Returns : RMSE

    """
    # Ensure both y and y_pred are 2D arrays with the same shape
    y = np.array(y).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)

    # Calculate RMSE
    return np.sqrt(((y - y_pred) ** 2).mean())


# Function to make table of cell proportions
def make_prop_table(adata: sc.AnnData, obs):
    """

    Makes proportion table from AnnData object's cell types.

    Parameters
    ----------
    adata: AnnData
    obs: str (name of obs with cell types)

    """
    num_cell_counter = Counter(adata.obs[obs])
    num_cells = []
    cell_types = []
    prop_cells = []
    tot_count = 0
    tot_prop = 0

    for cell in num_cell_counter:
        num_cells.append(num_cell_counter[cell])
        cell_types.append(cell)
        tot_count = tot_count + num_cell_counter[cell]

    for cell in num_cell_counter:
        proportion = num_cell_counter[cell] / tot_count
        prop_cells.append(proportion)
        tot_prop = tot_prop + proportion

    cell_types.append("Total")
    num_cells.append(tot_count)
    prop_cells.append(tot_prop)
    table = {"Cell_Types": cell_types, "Num_Cells": num_cells, "Prop_Cells": prop_cells}
    table = pd.DataFrame(table)
    return table
