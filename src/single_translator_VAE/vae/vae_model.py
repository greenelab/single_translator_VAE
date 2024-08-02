import logging
from typing import Optional

import numpy as np
import pandas as pd
import ray
import ray.train
import torch.cuda
import torch.nn as nn
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi._decorators import classproperty
from scvi._types import Tunable, TunableMixin
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
)
from scvi.dataloaders import DataSplitter
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass, RNASeqMixin, VAEMixin
from scvi.utils import setup_anndata_dsp
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from single_translator_VAE.vae.vae_module import VAEModule

logger = logging.getLogger(__name__)


class VAEModel(BaseModelClass, VAEMixin, TunableMixin, RNASeqMixin):
    """

    VAE Model for RNA-seq Translation.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~single_translator_VAE.vae_model.setup_anndata`.
    adata_obs
        AnnData observation key to be registered for training and testing
        (Example: for adata.obs["sex"] and adata.obs["data_type"], pass ["sex, "data_type"])
    labels_key
        Key for scVI to find obs and register: Optional, Default = "labels"
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    **model_kwargs
        Keyword args for :class:`~single_translator_VAE.VAEModule`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> single_translator_VAE.vae_model.setup_anndata(adata, batch_key="batch")
    >>> vae = single_translator_VAE.vae_model(adata)
    >>> vae.train()
    >>> adata.obsm["X_vae_model"] = vae.get_latent_representation()

    """

    __name__ = "Single-Nucleus - Single-Cell Translator VAE"  # Setting a class attribute

    def __init__(
        self,
        adata: AnnData,
        n_hidden: Tunable[int] = 610,
        n_latent: Tunable[int] = 460,
        n_layers: Tunable[int] = 3,
        batch_size: Tunable[int] = 150,
        use_batch_norm: Tunable[bool] = False,
        lr: Tunable[float] = 0.001,
        train_labels: Optional[list[str]] = None,
        kl_weight: Tunable[float] = 3,
        # class_weight: Tunable[float] = 100,
        recon_weight: Tunable[float] = 0.5,
        use_observed_lib_size: Tunable[bool] = False,
        dispersion: Tunable[str] = "gene",
        dropout_rate: Tunable[float] = 0.3,
        weight_decay: Tunable[float] = 0.001,
        encode_batch: Tunable[bool] = False,
        transform_batch: torch.Tensor | None = None,
        batch_representation: str = "one-hot",
        class_weights: bool = True,
        # kl_annealing_start: Tunable[float] = 10.0,
        # kl_annealing_steepness: Tunable[float] = 0.5,
        **model_kwargs,
    ):
        super().__init__(adata)
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lr = lr
        self.train_labels = train_labels
        self.adata = adata
        self.dispersion = dispersion
        self.dropout_rate = dropout_rate
        self.kl_weight = kl_weight
        # self.class_weight = class_weight
        self.recon_weight = recon_weight
        self.use_observed_lib_size = use_observed_lib_size
        self.n_cat_list = [self.adata.obs.labels_key.nunique()]
        self.weight_decay = weight_decay
        # self.kl_annealing_start = kl_annealing_start
        # self.kl_annealing_steepness = kl_annealing_steepness
        self.use_batch_norm = use_batch_norm
        self.encode_batch = encode_batch
        self.transform_batch = transform_batch
        self.batch_representation = batch_representation
        self.class_weights = class_weights

        library_log_means, library_log_vars = _init_library_size(self.adata_manager, self.summary_stats["n_batch"])

        # self.summary_stats provides information about anndata dimensions and other tensor info

        if self.class_weights:
            self.class_weights_tensor = self.calculate_class_weighting()

        self.module = VAEModule(
            n_input=self.summary_stats["n_vars"],
            n_hidden=self.n_hidden,
            n_latent=self.n_latent,
            n_layers=self.n_layers,
            lr=self.lr,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            batch_size=self.batch_size,
            adata=self.adata,
            kl_weight=self.kl_weight,
            # class_weight=self.class_weight,
            recon_weight=self.recon_weight,
            use_batch_norm=self.use_batch_norm,
            dispersion=self.dispersion,
            encode_batch=self.encode_batch,
            transform_batch=self.transform_batch,
            batch_representation=self.batch_representation,
            class_weights=self.class_weights,
            class_weights_tensor=self.class_weights_tensor if self.class_weights else None,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"Pytorch VAE Model: n_hidden: {n_hidden}, n_laten: {n_latent}, n_layers: {n_layers}, lr:{lr}"
        )
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("VAE model has been initialized")

        # and extracting labels:
        tl = pd.DataFrame(columns=["index", "label"])
        tl["index"], tl["label"] = range(0, len(self.adata.obs)), self.adata.obs.labels_key.values
        self.train_labels = tl.copy()

    def calculate_class_weighting(self):
        """
        Calculates class weights based on the inverse frequency of each class in the dataset.

        Returns
        -------
        torch.Tensor
            A tensor containing the weights for each class.
        """
        labels = self.adata.obs["labels_key"]
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        class_weights = total_samples / (len(unique_labels) * counts)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        return class_weights_tensor

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: Optional[str] = "labels_key",
        batch_key: Optional[str] = "batch",
        layer: Optional[str] = None,
        categorical_covariate_keys: Optional[list[str]] = None,
        continuous_covariate_keys: Optional[list[str]] = None,
        **kwargs,
    ) -> Optional[AnnDataManager]:
        """
        Register your AnnData for the model.

        Data Registration: The function registers the AnnData object with the model,
        configuring how data from AnnData will be used during the training of the VAE.
        This includes specifying which parts of the AnnData object contain relevant data like
        gene expression matrices, batch information, labels, and covariates.

        Parameters
        ----------
        adata : AnnData
            The AnnData object to preprocess.
        batch_key : str, optional
            Key in `adata.obs` for batch information.
        labels_key : list of str, optional
            Keys in `adata.obs` for labels.
        layer : str, optional
            Which layer to use for the model.
        categorical_covariate_keys : list of str, optional
            Additional categorical covariates.
        continuous_covariate_keys : list of str, optional
            Continuous covariates.

        Returns
        -------
        AnnData Manager Object, AnnData
        """
        assert not torch.isinf(torch.tensor(adata.X.toarray())).any(), "Input data contains infinite values"
        assert (torch.tensor(adata.X.toarray()) < 1e10).all(), "Input data contains extremely high values"
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

        return adata_manager, adata

    @classproperty
    def _tunables(cls):
        return [cls.__init__, cls.train]

    @classproperty
    def _metrics(cls):
        return ["validation_loss"]

    @staticmethod
    def shuffle_anndata(adata):
        """

        Return a shuffled version of the AnnData (across obserevations)

        Parameters
        ----------
        adata: AnnData

        Returns
        -------
        adata: AnnData (shuffled)

        """
        indices = np.random.permutation(adata.n_obs)
        return adata[indices].copy()

    @staticmethod
    def use_obs(
        adata: Optional[AnnData],
        adata_obs: str,
        labels_key: Optional[str] = "labels_key",
    ):
        """
        Create the observations adata_obs as a combined (or single) observation in adata["labels_key"]

        Parameters
        ----------
        adata: Optional[AnnData],
        adata_obs: str,
        labels_key: Optional[str] = "labels_key",
        --------------------------

        Returns
        -------
        adata: [AnnData] with labels_key as an observation with the merged adata_obs.
        -------

        """
        # Shuffle the AnnData object
        adata = VAEModel.shuffle_anndata(adata)

        if adata_obs is not None and isinstance(adata_obs, list):
            # Combine the multiple labels into one new column
            combined_vals = adata.obs[adata_obs].astype(str).agg("_".join, axis=1)
            adata.obs[labels_key] = combined_vals.values

        return adata

    def eval(self):
        """Set the module to evaluation mode."""
        self.module.eval()

    def train(
        self,
        max_epochs: int = 500,
        use_gpu: bool = False,
        validation_size: float = 0.2,
        patience: int = 18,
        min_delta: float = 0.35,
    ):
        """
        Train the model with validation support and a progress bar, with early stopping.

        Parameters
        ----------
        - max_epochs (int): Maximum number of training epochs.
        - use_gpu (bool): If True, train on GPU (if available).
        - validation_size (float): Fraction of data to use for validation.
        - patience (int): Number of epochs to wait for improvement before stopping.
        - min_delta (float): Minimum change to qualify as an improvement.

        Returns
        -------
        - train loss (mean)
        - validation loss (mean)
        - complete validation loss
        - complete train loss
        - train_losses (all)
        - val_losses (all)

        """
        # Split data into training and validation sets
        train_size = 1 - validation_size

        # Setup data loaders
        data_splitter = DataSplitter(
            self.adata_manager, train_size=train_size, validation_size=validation_size, batch_size=self.batch_size
        )
        data_splitter.setup()
        train_loader = data_splitter.train_dataloader()
        val_loader = data_splitter.val_dataloader()
        for batch in train_loader:
            assert not torch.isnan(batch[REGISTRY_KEYS.X_KEY]).any(), "Batch contains NaNs"

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.module.to(device)

        # Initialize optimizer
        optimizer = Adam(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Initialize learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=20, factor=0.85, min_lr=1e-5)

        # Progress bar setup
        progress_bar = tqdm(range(max_epochs), desc="Epochs", leave=True)

        complete_val_loss = []
        complete_train_loss = []
        train_losses = {key: [] for key in ["reconstruction_loss", "kl_local"]}  # , 'class_loss']}
        val_losses = {key: [] for key in ["reconstruction_loss", "kl_local"]}  # , 'class_loss']}

        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in progress_bar:
            epoch_train_losses = {key: [] for key in train_losses.keys()}
            epoch_val_losses = {key: [] for key in val_losses.keys()}
            train_loss, val_loss = 0, 0

            ## Training loop
            self.module.train()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                # Check for NaNs in input data
                if torch.isnan(batch[REGISTRY_KEYS.X_KEY]).any():
                    raise ValueError("Input data contains NaN values.")

                labels = batch[REGISTRY_KEYS.LABELS_KEY].long()
                batches_keys = batch[REGISTRY_KEYS.BATCH_KEY].long()

                # Perform model operations
                outputs = self.module.inference(batch[REGISTRY_KEYS.X_KEY], cat_list=labels, batch_index=batches_keys)

                # Check for NaNs in the outputs
                if torch.isnan(outputs["z"]).any() or torch.isnan(outputs["library"]).any():
                    raise ValueError("NaN values found in latent space or library size.")

                generative_outputs = self.module.generative(
                    z=outputs["z"], library=outputs["library"], cat_list=labels, batch_index=batches_keys
                )

                # Check for NaNs in the generative outputs
                if torch.isnan(generative_outputs["px_rate"]).any():
                    raise ValueError("NaN values found in generative outputs.")

                # Compute loss
                loss_output, loss_dict = self.module.loss(batch, outputs, generative_outputs)
                loss = loss_output.loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=0.85)  # Gradient clipping
                optimizer.step()
                train_loss += loss.item()  # Accumulate total loss for averaging later
                for key in epoch_train_losses:
                    epoch_train_losses[key].append(loss_dict[key])

            ## Validation loop
            self.module.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch[REGISTRY_KEYS.LABELS_KEY].long()
                    batches_keys = batch[REGISTRY_KEYS.BATCH_KEY].long()

                    outputs = self.module.inference(
                        batch[REGISTRY_KEYS.X_KEY], cat_list=labels, batch_index=batches_keys
                    )

                    # Check for NaNs in the outputs during validation
                    if torch.isnan(outputs["z"]).any() or torch.isnan(outputs["library"]).any():
                        raise ValueError("NaN values found in latent space or library size during validation.")

                    generative_outputs = self.module.generative(
                        z=outputs["z"], library=outputs["library"], cat_list=labels, batch_index=batches_keys
                    )

                    # Check for NaNs in the generative outputs during validation
                    if torch.isnan(generative_outputs["px_rate"]).any():
                        raise ValueError("NaN values found in generative outputs during validation.")

                    loss_output, loss_dict = self.module.loss(batch, outputs, generative_outputs)
                    val_loss += loss_output.loss.item()
                    for key in epoch_val_losses:
                        epoch_val_losses[key].append(loss_dict[key])

            # Calculate mean losses for the epoch
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            complete_train_loss.append(train_loss)
            complete_val_loss.append(val_loss)

            for key in train_losses:
                train_losses[key].append(np.mean(epoch_train_losses[key]))  # Average per epoch
            for key in val_losses:
                val_losses[key].append(np.mean(epoch_val_losses[key]))

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Update progress bar
            progress_bar.set_description(
                f"Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            ray.train.report({"validation_loss": val_loss})

            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        self.is_trained = True
        return train_loss, val_loss, complete_val_loss, complete_train_loss, train_losses, val_losses

    @classproperty
    def _metrics(cls):
        return ["validation_loss"]

    # KEEPING THIS FUNCT FOR NOW, MIGHT STILL ADD ANNEALING LATER.

    # def kl_annealing(self,
    #                 epoch,
    #                 midpoint,
    #                 annealing_start):
    #     """ Sigmoid KL Annealing Function
    #     Parameters:
    #     - epoch (int): Current epoch number.
    #     - mid_point (int): The epoch at which the sigmoid function is at its midpoint.
    #     - steepness (float): Controls the steepness of the sigmoid curve (larger values make the transition sharper).

    #     Returns:
    #     - float: Current KL weight.
    #     """
    #     if epoch < annealing_start:
    #         val=0.0
    #     else:
    #         val=1 / (1 + np.exp(-self.kl_annealing_steepness * (epoch - midpoint)))
    #     return val
