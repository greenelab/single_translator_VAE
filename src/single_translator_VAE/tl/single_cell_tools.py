import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import ray as ray
import scanpy as sc
import sklearn as sk
import torch
import torch.nn
from scipy.optimize import nnls
from scipy.stats import poisson
from sklearn.preprocessing import MinMaxScaler

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


def prep_files_for_deconvolution(bulks, reference, path, file_name, deconvolution_method):
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

    method_path = os.path.join(path, deconvolution_method, "input_data")
    if not os.path.exists(method_path):
        os.makedirs(method_path)

    files = {}
    match deconvolution_method:
        case "cibersortx":
            reference_data = np.vstack([np.append("gene_ids", cell_types), np.column_stack([genes, reference])])
            ref_file_path = os.path.join(method_path, f"{file_name}_reference.txt")
            np.savetxt(ref_file_path, reference_data, fmt="%s", delimiter="\t")
            files["reference"] = ref_file_path

            if bulks is not None:
                bulks_data = np.vstack(
                    [np.append("gene_ids", np.arange(bulks.shape[1])), np.column_stack([genes, bulks])]
                )
                bulks_file_path = os.path.join(method_path, f"{file_name}_bulks.txt")
                np.savetxt(bulks_file_path, bulks_data, fmt="%s", delimiter="\t")
                files["bulks"] = bulks_file_path

        case "nnls":
            # Making cell type reference, then scaling
            ref_raw = pd.DataFrame(index=genes, columns=cell_types)
            for cell_type in cell_types:
                cell_df = reference[reference.obs["cell_types"].isin([cell_type])].X.toarray()
                cell_sample = sk.utils.resample(cell_df, n_samples=10000, replace=True)
                ref_raw[cell_type] = cell_sample.sum(axis=0)

            # clippign before scaling to 95th pecentile
            clip_upper = np.quantile(ref_raw.values, 0.95)
            ref_raw_val = np.clip(ref_raw.values, 0, clip_upper)

            # and scaling to be between values 0 and 1 to use for NNLS
            scaler = MinMaxScaler()
            scaler.fit(ref_raw_val)
            ref_raw_val = scaler.transform(ref_raw_val)
            reference_data = pd.DataFrame(ref_raw_val, index=genes, columns=cell_types)

            ref_file_path = os.path.join(method_path, f"{file_name}_reference.txt")
            np.savetxt(ref_file_path, reference_data, fmt="%s", delimiter="\t")
            files["reference"] = reference_data

            if bulks is not None:
                # clippign before scaling to 95th pecentile
                clip_upper = np.quantile(bulks.values, 0.95)
                pseudo_df = np.clip(bulks.values, 0, clip_upper)
                # and normalize to values between 0 and 1
                scaler = MinMaxScaler()
                scaler.fit(pseudo_df)
                normalized_pseudo_df = scaler.transform(pseudo_df)
                bulks_data = pd.DataFrame(normalized_pseudo_df, columns=genes)

                bulks_file_path = os.path.join(method_path, f"{file_name}_bulks.txt")
                np.savetxt(bulks_file_path, bulks_data, fmt="%s", delimiter="\t")
                files["bulks"] = bulks_data

        case "bayesprism":
            reference_df = pd.DataFrame(reference, index=genes, columns=cell_types)
            ref_file_path = os.path.join(method_path, f"{file_name}_reference.csv")
            reference_df.to_csv(ref_file_path, index_label=False)
            files["reference"] = ref_file_path

            if bulks is not None:
                bulks_df = pd.DataFrame(bulks, index=genes, columns=np.arange(bulks.shape[1]))
                bulks_file_path = os.path.join(method_path, f"{file_name}_bulks.csv")
                bulks_df.to_csv(bulks_file_path, index_label=False)
                files["bulks"] = bulks_file_path

    return files


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


def calc_nnls(all_refs, prop_df, pseudo_df):
    """
    Perform non-negative least squares (NNLS) deconvolution of simulated bulk RNA-seq data.

    This function applies NNLS to deconvolve simulated bulk RNA-seq data using different reference datasets.
    It compares the predicted cell type proportions with known proportions and calculates residuals.

    Parameters
    ----------
    all_refs : dict of pd.DataFrame
        A dictionary containing reference datasets for deconvolution. Each key is a string identifying the reference type,
        and each value is a DataFrame where rows correspond to cell types and columns to features (e.g., genes).
    prop_df : pd.DataFrame
        DataFrame containing the known proportions of cell types for each sample in the simulated bulk data.
        Each row corresponds to a sample, and each column corresponds to a cell type.
    pseudo_df : pd.DataFrame
        DataFrame containing the simulated bulk RNA-seq data. Each row corresponds to a sample,
        and each column corresponds to a feature (e.g., gene expression levels).

    Returns
    -------
    calc_prop_tot : dict of pd.DataFrame
        A dictionary containing the estimated proportions of cell types for each reference.
        Each key corresponds to a reference type, and each value is a DataFrame where rows are samples and columns are cell types.
    calc_res_tot : dict of np.ndarray
        A dictionary containing the residuals (error terms) from the NNLS fitting for each reference.
        Each key corresponds to a reference type, and each value is an array of residuals for each sample.
    custom_res_tot : dict of pd.DataFrame
        A dictionary containing the residuals between the known and estimated cell type proportions for each reference.
        Each key corresponds to a reference type, and each value is a DataFrame where rows are samples and columns are cell types.
    comparison_prop_tot : dict
        A placeholder dictionary for additional comparison results, currently not implemented.
    missing_cell_tot : dict
        A placeholder dictionary for handling missing cell types, currently not implemented.

    Notes
    -----
    - Ensure that the reference datasets (`all_refs`) and the known proportions (`prop_df`) have consistent dimensions.
    """
    calc_prop_tot = {}
    calc_res_tot = {}
    custom_res_tot = {}

    for exp, ref_df in all_refs.items():
        calc_prop_all = pd.DataFrame()
        custom_res_all = pd.DataFrame()
        calc_res_all = []

        print(f"Reference: {exp}")

        # Extracting reference matrix and verifying its integrity
        ref = ref_df.values
        if ref.shape[1] != prop_df.shape[1]:
            raise ValueError(f"Reference '{exp}' and prop_df have inconsistent dimensions.")

        # Calculate predicted values and residuals for each row
        for sample in range(len(pseudo_df)):
            sample_data = pseudo_df.iloc[sample].values

            # Apply NNLS to obtain the estimated proportions
            calc_prop, calc_res = nnls(ref, sample_data)

            # Normalize proportions
            tot = np.sum(calc_prop)
            if tot > 0:
                calc_prop = calc_prop / tot
            else:
                calc_prop = np.zeros_like(calc_prop)

            # Compute residuals compared to known proportions
            actual_prop = prop_df.iloc[sample].values
            custom_res = actual_prop - calc_prop

            # Convert to DataFrame for easier handling
            calc_prop_df = pd.DataFrame(calc_prop).T
            custom_res_df = pd.DataFrame(custom_res).T

            # Append to accumulated DataFrames
            calc_prop_all = pd.concat([calc_prop_all, calc_prop_df], ignore_index=True)
            custom_res_all = pd.concat([custom_res_all, custom_res_df], ignore_index=True)
            calc_res_all.append(calc_res)

        # Store results in the respective dictionaries
        calc_prop_tot[exp] = calc_prop_all
        calc_prop_tot[exp].columns = ref_df.columns
        calc_res_tot[exp] = np.array(calc_res_all)
        custom_res_tot[exp] = custom_res_all

    return calc_prop_tot, calc_res_tot, custom_res_tot


def evaluate_deconvolution_references(calc_prop_tot, prop_df):
    """
    Evaluate the performance of different reference datasets using RMSE and Pearson's correlation.

    This function calculates the Root Mean Squared Error (RMSE) and Pearson's correlation coefficient
    between the estimated and real cell type proportions for each reference dataset.

    Parameters
    ----------
    calc_prop_tot : dict of pd.DataFrame
        A dictionary containing the estimated proportions of cell types for each reference.
        Each key corresponds to a reference type, and each value is a DataFrame where rows are samples and columns are cell types.
    prop_df : pd.DataFrame
        DataFrame containing the known proportions of cell types for each sample. Each row corresponds to a sample, and each column corresponds to a cell type.

    Returns
    -------
    evaluation_results : dict
        A dictionary where each key corresponds to a reference type and each value is another dictionary containing
        the RMSE and Pearson's correlation coefficient for that reference.
    """
    evaluation_results = {}

    for ref_name, est_prop_df in calc_prop_tot.items():
        # Ensure the indices and columns match
        est_prop_df = est_prop_df.reindex(prop_df.index)
        est_prop_df = est_prop_df[prop_df.columns]

        # Calculate RMSE using the custom rmse function
        rmse_value = rmse(prop_df.values, est_prop_df.values)

        # Calculate Pearson's correlation
        correlation = np.corrcoef(prop_df.values.flatten(), est_prop_df.values.flatten())[0, 1]

        evaluation_results[ref_name] = {"RMSE": rmse_value, "Correlation": correlation}

    return evaluation_results
