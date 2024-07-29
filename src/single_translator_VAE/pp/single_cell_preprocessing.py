import gzip
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
import torch
from anndata import AnnData, read_h5ad
from sklearn.utils import resample

import single_translator_VAE as sv


def validate_data(adata):
    """
    Funct. to check NaNs in data.

    Parameters
    ----------
    adata: AnnData Object

    This will check the adata doesn't have any NaNs

    """
    if np.isnan(adata.X.toarray()).any():
        raise ValueError("Input data contains NaNs")
    return adata


def load_PNB_data(data_type: str, load_testing: bool = False):
    """
    Open the data of pediatric neuroblastoma. We have 1 SN and 1 SC from same patient, 1 SC and SN others for testing.

    Parameters
    ----------
    data_type : str (either "single_nucleus" or "single_cell")
    load_testing : bool (True /  False) same patient is True.

    Returns
    -------
    AnnData
        TheAnnData object.
    Metadata
        Pandas Dataframe.

    """
    res_name = "PNB"
    adata = []
    meta_data = []
    path = f"{os.getcwd()}/../../data/{res_name}/"  # Path to original data.

    if data_type == "single_cell":
        if load_testing:
            # File paths
            h5_gz_file = Path(path, "GSM4186962_HTAPP-312-SMP-902_fresh-C4-T2_channel1_raw_gene_bc_matrices_h5.h5.gz")
            h5_file = Path(path, "GSM4186962_HTAPP-312-SMP-902_fresh-C4-T2_channel1_raw_gene_bc_matrices_h5.h5")
            csv_file = Path(path, "GSM4186962_metadata_HTAPP-312-SMP-902_fresh-C4-T2_channel1.csv.gz")

        else:
            # File paths
            h5_gz_file = Path(path, "GSM4186963_HTAPP-656-SMP-3481_fresh-T1_channel1_raw_gene_bc_matrices_h5.h5.gz")
            h5_file = Path(path, "GSM4186963_HTAPP-656-SMP-3481_fresh-T1_channel1_raw_gene_bc_matrices_h5.h5")
            csv_file = Path(path, "GSM4186963_metadata_HTAPP-656-SMP-3481_fresh-T1_channel1.csv.gz")

        # Decompressing the h5.gz file
        with gzip.open(h5_gz_file, "rb") as f_in:
            with open(h5_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Importing the data
        sc_adata = sc.read_10x_h5(h5_file)  # Read the h5 file
        sc_adata.var_names_make_unique()
        # Read the CSV file using pandas
        sc_metadata = pd.read_csv(csv_file)
        # clean up cell name to match obs in AnnData
        sc_metadata["cell"] = sc_metadata["Unnamed: 0"].apply(sv.tl.split_ID_2).astype(str)
        sc_metadata["cell"] = sc_metadata["cell"].apply(sv.tl.merge_strings)
        adata = sc_adata
        meta_data = sc_metadata

    elif data_type == "single_nucleus":
        if load_testing:
            # File paths
            h5_gz_file = Path(path, "GSM4186968_HTAPP-244-SMP-451_TST_channel1_raw_gene_bc_matrices_h5.h5.gz")
            h5_file = Path(path, "GSM4186968_HTAPP-244-SMP-451_TST_channel1_raw_gene_bc_matrices_h5.h5")
            csv_file = Path(path, "GSM4186968_metadata_HTAPP-244-SMP-451_TST_channel1.csv.gz")

        else:
            # File paths
            h5_gz_file = Path(path, "GSM4186969_HTAPP-656-SMP-3481_TST_channel1_raw_gene_bc_matrices_h5.h5.gz")
            h5_file = Path(path, "GSM4186969_HTAPP-656-SMP-3481_TST_channel1_raw_gene_bc_matrices_h5.h5")
            csv_file = Path(path, "GSM4186969_metadata_HTAPP-656-SMP-3481_TST_channel1.csv.gz")

        # Decompressing the h5.gz file
        with gzip.open(h5_gz_file, "rb") as f_in:
            with open(h5_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Importing the data
        sn_adata = sc.read_10x_h5(h5_file)  # Read the h5 file
        sn_adata.var_names_make_unique()
        # Read the CSV file using pandas
        sn_metadata = pd.read_csv(csv_file)
        # clean up cell name to match obs
        sn_metadata["cell"] = sn_metadata["Unnamed: 0"].apply(sv.tl.split_ID_2).astype(str)
        sn_metadata["cell"] = sn_metadata["cell"].apply(sv.tl.merge_strings)
        adata = sn_adata
        meta_data = sn_metadata

    else:
        print('Give valid data type: "single_cell" or "single_nucleus"')

    return adata, meta_data


def load_MBC_data(data_type: str, load_testing: bool = False):
    """
    Open the data og MBC. We have 1 SN and 1 SC from same patient, and 1 SC from another for testing.

    Parameters
    ----------
    data_type : str (either "single_nucleus" or "single_cell")

    Returns
    -------
    AnnData
        TheAnnData object.
    Metadata
        Pandas Dataframe.
    """
    res_name = "MBC"
    adata = []
    meta_data = []
    path = f"{os.getcwd()}/../../data/{res_name}/"  # Path to original data.

    if data_type == "single_cell":
        if load_testing:
            # File paths
            h5_gz_file = Path(path, "GSM4186973_HTAPP-285-SMP-751_fresh_channel1_raw_gene_bc_matrices_h5.h5.gz")
            h5_file = Path(path, "GSM4186973_HTAPP-285-SMP-751_fresh_channel1_raw_gene_bc_matrices_h5.h5")
            csv_file = Path(path, "GSM4186973_metadata_HTAPP-285-SMP-751_fresh_channel1.csv.gz")

            # Decompressing the h5.gz file
            with gzip.open(h5_gz_file, "rb") as f_in:
                with open(h5_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Importing the data
            sc_adata = sc.read_10x_h5(h5_file)  # Read the h5 file
            sc_adata.var_names_make_unique()
            # Read the CSV file using pandas
            sc_metadata = pd.read_csv(csv_file)
            # clean up cell name to match obs in AnnData
            sc_metadata["cell"] = sc_metadata["Unnamed: 0"].apply(sv.tl.split_ID).astype(str)
            sc_metadata["cell"] = sc_metadata["cell"].apply(sv.tl.merge_strings)

            adata = sc_adata
            meta_data = sc_metadata

        else:
            # File paths
            h5_gz_file = Path(path, "GSM4186974_HTAPP-963-SMP-4741_fresh_channel1_raw_gene_bc_matrices_h5.h5.gz")
            h5_file = Path(path, "GSM4186974_HTAPP-963-SMP-4741_fresh_channel1_raw_gene_bc_matrices_h5.h5")
            csv_file = Path(path, "GSM4186974_metadata_HTAPP-963-SMP-4741_fresh_channel1.csv.gz")

            # Decompressing the h5.gz file
            with gzip.open(h5_gz_file, "rb") as f_in:
                with open(h5_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Importing the data
            sc_adata = sc.read_10x_h5(h5_file)  # Read the h5 file
            sc_adata.var_names_make_unique()
            # Read the CSV file using pandas
            sc_metadata = pd.read_csv(csv_file)
            # clean up cell name to match obs in AnnData
            sc_metadata["cell"] = sc_metadata["Unnamed: 0"].apply(sv.tl.split_ID).astype(str)
            sc_metadata["cell"] = sc_metadata["cell"].apply(sv.tl.merge_strings)

            adata = sc_adata
            meta_data = sc_metadata

    elif data_type == "single_nucleus":
        # File paths
        h5_gz_file = Path(path, "GSM4186980_HTAPP-963-SMP-4741_TST_channel1_raw_gene_bc_matrices_h5.h5.gz")
        h5_file = Path(path, "GSM4186980_HTAPP-963-SMP-4741_TST_channel1_raw_gene_bc_matrices_h5.h5")
        csv_file = Path(path, "GSM4186980_metadata_HTAPP-963-SMP-4741_TST_channel1.csv.gz")

        # Decompressing the h5.gz file
        with gzip.open(h5_gz_file, "rb") as f_in:
            with open(h5_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Importing the data
        sn_adata = sc.read_10x_h5(h5_file)  # Read the h5 file
        sn_adata.var_names_make_unique()
        # Read the CSV file using pandas
        sn_metadata = pd.read_csv(csv_file)
        # clean up cell name to match obs
        sn_metadata["cell"] = sn_metadata["Unnamed: 0"].apply(sv.tl.split_ID).astype(str)
        sn_metadata["cell"] = sn_metadata["cell"].apply(sv.tl.merge_strings)

        adata = sn_adata
        meta_data = sn_metadata

    else:
        print('Give valid data type: "single_cell" or "single_nucleus"')

    return adata, meta_data


def assign_cell_types(adata: AnnData, cell_types_assign: np.array) -> AnnData:
    """
    Assign cell types from DataFrame in obs on the AnnData object and make sure the same cell-types are in both.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to assign cell types to.
    cell_types: Numpy Array
        The cell types to assign.

    Returns
    -------
    AnnData
        The AnnData object with cell types assigned in .obs["cell_types"].
    """
    # Check if the length matches:
    if len(cell_types_assign) != adata.X.shape[0]:
        raise AssertionError("Cell types do not match in Anndata and NumPy array")

    adata.obs["cell_types"] = cell_types_assign

    print(f"Done!\nCell Types in Data are: {adata.obs.cell_types.value_counts()}")

    return adata


def filter_with_miQC(adata: AnnData, base_name: str) -> AnnData:
    """
    Run miQC for preprocessing on the AnnData object.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to preprocess using miQC.
    base_dir : str
        Name of data to use in /data/ directory (/data/{base_name}_data.h5ad").

    Returns
    -------
    AnnData
        The processed AnnData object.
    """
    # Directory for R scripts and data
    # Get the directory of the current Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_script_path = os.path.join(script_dir, "R", "filter_miQC.R")
    input_path = os.path.join(script_dir, "..", "..", "..", "data", f"{base_name}_data.h5ad")
    output_path = os.path.join(script_dir, "..", "..", "..", "data", f"{base_name}_data_processed.h5ad")

    # Check if the R script exists
    if not os.path.exists(r_script_path):
        raise FileNotFoundError(f"R script not found at {r_script_path}")

    # Check if counts are in the main matrix, if not, attempt to find them or set the main matrix as counts
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X  # assuming X holds raw counts

    # Write input data
    adata.write_h5ad(input_path)

    # Run the R script
    print("Processing with miQC...\n...")
    ro.r["source"](r_script_path)
    apply_miQC = ro.globalenv["apply_miQC"]
    apply_miQC(input_path, output_path)

    # Read the processed data
    processed_adata = read_h5ad(output_path)
    print("\nDone!")
    return processed_adata


def match_cell_types(adata1: AnnData, adata2: AnnData, cells_to_keep: np.array):
    """
    Match cell types in 2 AnnData objects according to cells in cells_to_keep.

    Parameters
    ----------
    adata1 : AnnData
        The AnnData object to match cell types.
    adata2 : AnnData
        The AnnData object to match cell types.
    cell_types_to_keep: Numpy Array
        The cell types to keep in both Anndatas

    Returns
    -------
    adata1:
        AnnData with same cell types.
    adata2:
        AnnData with same cell types.
    """
    # match cells:
    adata1 = adata1[adata1.obs["cell_types"].isin(cells_to_keep)]
    adata2 = adata2[adata2.obs["cell_types"].isin(cells_to_keep)]

    return adata1, adata2


def split_test_train(adata: AnnData, percentage_test: float) -> AnnData:
    """
    Split the AnnData object into testing and training sets based on cell types, data type, and sex.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to split into train and test cells.
    percentage_test:
        Float, percetage of data to be in Testing set.

    Returns
    -------
    AnnData
        The split AnnData object with .obs["TrainTest"].
    """
    # Generate keys for grouping
    first = adata.obs["cell_types"].astype(str) + adata.obs["data_type"].astype(str)
    second = adata.obs["sex"].astype(str) + adata.obs["patient_id"].astype(str)
    adata.obs["test_train_key"] = first + second

    # Initialize the labels
    adata.obs["TrainTest"] = "Train"

    for curr_key in adata.obs["test_train_key"].unique():
        curr_idx = np.where(adata.obs["test_train_key"] == curr_key)[0]
        num_test = int(len(curr_idx) * percentage_test)

        # Random selection of indices for test and unknown labels
        test_idx = np.random.choice(curr_idx, num_test, replace=False)

        adata.obs["TrainTest"].iloc[test_idx] = "Test"

    return adata


def prepare_data_and_labels(
    adata,
    labels_key="labels_key",
    label_map: Optional[dict] = None,
):
    """
    Prepare data and labels with correct encoding even if one label type is present.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labels_key : str
        Column in adata.obs to use for labels.
    label_map: dict
        Examaple: {'female_single_cell': 0, 'female_single_nucleus': 1}:


    Returns
    -------
    input_tensor : torch.Tensor
        Tensor of input data.
    labels_tensor : torch.Tensor
        Tensor of encoded labels.
    ]
    """
    if label_map is None:
        label_map = {"single_cell": 0, "single_nucleus": 1}
    batch_tensor = []

    # Ensure all labels are strings
    adata.obs[labels_key] = adata.obs[labels_key].astype(str)

    # Apply label map and handle missing labels by mapping with get to avoid KeyErrors
    labels_encoded = adata.obs[labels_key].map(lambda x: label_map.get(x, -1)).values

    # Check for any labels that did not match the expected keys
    if np.any(labels_encoded == -1):
        print("Warning: Some labels did not match the expected keys and were set to -1")

    labels_tensor = torch.tensor(labels_encoded, dtype=torch.long).unsqueeze(1)

    # Prepare input data
    input_data = adata.X.toarray()
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Prepare batch information
    if "batch" in adata.obs.columns:
        batch_tensor = torch.tensor(adata.obs["batch"].astype("category").cat.codes.values, dtype=torch.long).unsqueeze(
            1
        )
    else:
        batch_tensor = None

    return input_tensor, labels_tensor, batch_tensor


def create_training_data(original_adata, percentage_real=33):
    """
    Create more training data by adding Gaussian Noise to original.

    Parameters
    ----------
    - original_adata: AnnData object with the original data.
    - percentage_real: Integer, percentage of real data to use.

    Returns
    -------
    - An AnnData object with the new training data including the synthetic data.
    """
    how_many = int(100 * original_adata.obs.shape[0] / percentage_real)
    original_adata.obs.index = range(0, len(original_adata))
    real_data = original_adata.X.toarray()
    real_obs = original_adata.obs

    # Resample indices
    resampled_indices = resample(original_adata.obs.index, replace=True, n_samples=how_many)

    # Get the resampled data and obs
    sampled_data = real_data[resampled_indices]
    sampled_obs = real_obs.iloc[resampled_indices]

    # Add Gaussian noise
    noise = np.random.normal(1, 0.05, sampled_data.shape)
    noisy_sampled_data = sampled_data * noise
    noisy_sampled_data = np.clip(noisy_sampled_data, a_min=0, a_max=None)

    # Set values between 0 and 1 to 0 in noisy sampled data
    noisy_sampled_data = np.where((noisy_sampled_data > 0) & (noisy_sampled_data < 1), 0, noisy_sampled_data)

    # Create the new AnnData object
    synthetic_adata = sc.AnnData(X=noisy_sampled_data, obs=sampled_obs, var=original_adata.var)
    # # Convert all values to integers
    # synthetic_adata.X = synthetic_adata.X.astype(int)

    # Combine the original and synthetic data
    combined_adata = sc.concat([original_adata, synthetic_adata], axis=0)
    combined_adata.layers["counts"] = combined_adata.X.copy()
    combined_adata.obs_names_make_unique()
    return combined_adata
