import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi._decorators import classproperty
from scvi._types import Tunable, TunableMixin
from scvi.distributions import NegativeBinomial
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
)
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch.distributions import Distribution, Normal
from torch.nn.functional import linear

TensorDict = dict[str, torch.Tensor]


class VAEModule(BaseModuleClass, TunableMixin):
    """
    VAE Module for single-cell data.

    Parameters
    ----------
    n_input
        Number of input genes
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        n_input: int,
        library_log_means: np.ndarray,
        library_log_vars: np.ndarray,
        n_hidden: int = 610,
        n_latent: int = 460,
        n_layers: int = 2,
        lr: float = 0.002,
        adata: AnnData = None,
        kl_weight: Tunable[float] = 3,
        recon_weight: Tunable[float] = 0.5,
        dispersion: str = "gene",
        use_observed_lib_size: bool = False,
        dropout_rate: Tunable[float] = 0.3,
        use_batch_norm: bool = False,
        transform_batch: torch.Tensor | None = None,
        encode_batch: bool = False,
        batch_representation: str = "one-hot",
        **kwargs,
    ):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.adata = adata
        self.use_batch_norm = use_batch_norm
        self.n_batch = adata.obs["batch"].nunique()
        self.n_cat_list = [adata.obs["labels_key"].nunique()]
        self.dropout_rate = dropout_rate
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.dispersion = dispersion
        self.transform_batch = transform_batch
        self.n_labels = adata.obs["labels_key"].nunique()
        self.use_observed_lib_size = use_observed_lib_size
        self.encode_batch = encode_batch
        self.n_input = n_input + self.n_batch if encode_batch else n_input
        self.n_latent = n_latent
        self.transform_batch = transform_batch
        self.batch_representation = batch_representation

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, self.n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError("dispersion must be one of 'gene', 'gene-batch', 'gene-label', 'gene-cell'.")
        # self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": self.n_latent,
            "n_layers": n_layers,
            "lr": lr,
            "dropout_rate": self.dropout_rate,
            **kwargs,
        }

        # this is needed to comply with some requirement of the VAEMixin class,
        # does not affect out NB distribtuion at decoding.
        self.latent_distribution = "normal"

        if not self.use_observed_lib_size:
            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # This encoder contains 3 encoders:
        # one for x to z, one for mean (linear) and one for var (linear).
        self.z_encoder = Encoder(
            self.n_input,
            n_cat_list=self.n_cat_list,
            n_output=self.n_latent,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            inject_covariates=True,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=False,
            dropout_rate=self.dropout_rate,
        )

        # l encoder goes from n_input-dimensional data to 1-d library size
        # This encoder contains 3 encoders:
        # one for x to library size, one for mean (linear) and one for var (linear) of library.
        self.l_encoder = Encoder(
            self.n_input,
            n_output=1,
            n_cat_list=self.n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            inject_covariates=True,
            dropout_rate=self.dropout_rate,
        )

        # decoder goes from n_latent-dimensional space to n_input-d data
        # This decoder includes 4 decoders:
        # one for px (z to new x)
        # one for px_r (dispersion) (linear decoder) - only if dispersion == "gene-cell"
        #       else, px_r is an optimizable parameter
        # one for px_dropout (like variance) (linear decoder)
        # one for px_scale, related to dispersion, not linear, used for px_rate, scaled by library.
        # px_rate = torch.exp(library) * px_scale
        self.decoder = DecoderSCVI(
            n_input=self.n_latent + (self.n_batch if encode_batch else 0),
            n_output=self.n_input - (self.n_batch if encode_batch else 0),
            n_cat_list=self.n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_layer_norm=False,
            use_batch_norm=self.use_batch_norm,
        )

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def inference(self, x: torch.Tensor, cat_list: torch.Tensor, batch_index: torch.Tensor):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        assert not torch.isnan(x).any(), "Input contains NaNs"

        # log the input to the variational distribution for numerical stability
        x_ = torch.log(1 + x)

        assert not torch.isnan(x_).any(), "Log transformation produced NaNs"

        if self.encode_batch:
            if self.transform_batch is not None:
                batch_index = torch.ones_like(batch_index) * self.transform_batch
            if self.batch_representation == "embedding":
                raise ValueError("embedding of batch information not currently available")
            elif self.batch_representation == "one-hot":
                x_ = torch.cat([x_, one_hot(batch_index, self.n_batch).float()], dim=-1)
            else:
                raise ValueError("batch_representation must be one of 'one-hot', 'embedding'.")

        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
            ql_m, ql_v = None, None
            assert not torch.isnan(library).any(), "Library size calculation produced NaNs"
        else:
            ql_m, ql_v, library = self.l_encoder(x_, cat_list)
            assert not torch.isnan(ql_m).any(), "ql_m contains NaNs"
            assert not torch.isnan(ql_v).any(), "ql_v contains NaNs"
            assert not torch.isnan(library).any(), "Library size calculation produced NaNs"

        # get variational parameters via the encoder networks
        qz_m, qz_v, z = self.z_encoder(x_, cat_list)
        assert not torch.isnan(qz_m).any(), "qz_m contains NaNs"
        assert not torch.isnan(qz_v).any(), "qz_v contains NaNs"
        assert not torch.isnan(z).any(), "z contains NaNs"

        outputs = {"z": z, "qz_m": qz_m, "qz_v": qz_v, "ql_m": ql_m, "ql_v": ql_v, "library": library}

        return outputs

    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        cat_list: torch.Tensor,
        batch_index: torch.Tensor,
    ):
        """Runs the generative model."""
        if self.encode_batch:
            if self.transform_batch is not None:
                batch_index = torch.ones_like(batch_index) * self.transform_batch
            if self.batch_representation == "embedding":
                raise ValueError("embedding of batch information not currently available")

            elif self.batch_representation == "one-hot":
                z = torch.cat([z, one_hot(batch_index, self.n_batch).float()], dim=-1)
            else:
                raise ValueError("batch_representation must be one of 'one-hot', 'embedding'.")

        px_scale, px_r, px_rate, _ = self.decoder(self.dispersion, z, library, cat_list)
        if self.dispersion == "gene-label":
            # px_r = linear(
            #    one_hot(cat_list.squeeze(-1), self.n_labels).float(), self.px_r
            # )
            px_r = linear(one_hot(cat_list, self.n_labels).float(), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        outputs = {"px_scale": px_scale, "px_r": px_r, "px_rate": px_rate}

        return outputs

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
    ) -> LossOutput:
        """Compute the loss."""
        from torch.distributions import kl_divergence

        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        # KL divergence for the latent space
        kl_divergence_z = kl_divergence(
            Normal(inference_outputs["qz_m"], torch.sqrt(inference_outputs["qz_v"])),
            Normal(torch.zeros_like(inference_outputs["qz_m"]), torch.ones_like(inference_outputs["qz_v"])),
        ).sum(dim=-1)

        # KL divergence for the library size
        if not self.use_observed_lib_size:
            local_library_log_means, local_library_log_vars = self._compute_local_library_params(batch_index)
            kl_divergence_l = kl_divergence(
                Normal(inference_outputs["ql_m"], torch.sqrt(inference_outputs["ql_v"])),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        # Reconstruction loss
        reconst_loss = (
            -NegativeBinomial(mu=generative_outputs["px_rate"], theta=generative_outputs["px_r"]).log_prob(x).sum(-1)
        )

        weighted_kl = self.kl_weight * (kl_divergence_z + kl_divergence_l)
        weighted_recon = self.recon_weight * reconst_loss

        # Combine all weighted losses
        total_loss = torch.mean(weighted_recon + weighted_kl)
        losses_dict = {
            "reconstruction_loss": weighted_recon.mean().item(),
            "kl_local": weighted_kl.mean().item(),
        }

        return LossOutput(
            loss=total_loss,
            reconstruction_loss=weighted_recon,
            kl_local={"kl_divergence_z": kl_divergence_z, "kl_divergence_l": kl_divergence_l},
        ), losses_dict

    def _compute_local_library_params(self, batch_index: torch.Tensor):
        """Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        from torch.nn.functional import linear

        n_batch = self.library_log_means.shape[1]

        local_library_log_means = linear(one_hot(batch_index, n_batch).float(), self.library_log_means)

        local_library_log_vars = linear(one_hot(batch_index, n_batch).float(), self.library_log_vars)

        return local_library_log_means, local_library_log_vars

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ) -> torch.Tensor:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:p(\hat{x} \mid x).

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to

        Returns
        -------
        x_new
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = n_samples = n_samples
        (
            _,
            generative_outputs,
        ) = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]

        dist = NegativeBinomial(mu=px_rate, theta=px_r)

        if n_samples > 1:
            exprs = dist.sample().permute([1, 2, 0])  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()

    @torch.no_grad()
    def marginal_ll(self, tensors: TensorDict, n_mc_samples: int, **kwargs):
        """Marginal ll, updated to handle unexpected keyword arguments gracefully."""
        sample_batch = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(tensors)
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            z = inference_outputs["z"]
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            reconst_loss = losses.dict_sum(losses.reconstruction_loss)

            # Log-probabilities
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(one_hot(batch_index, n_batch), self.library_log_means)
            local_library_log_vars = F.linear(one_hot(batch_index, n_batch), self.library_log_vars)
            p_l = Normal(local_library_log_means, local_library_log_vars.sqrt()).log_prob(library).sum(dim=-1)

            p_z = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)).log_prob(z).sum(dim=-1)
            p_x_zl = -reconst_loss
            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(library).sum(dim=-1)

            to_sum[:, i] = p_z + p_l + p_x_zl - q_z_x - q_l_x

        batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl

    @classproperty
    def _tunables(cls):
        return [
            cls.__init__,
            cls._get_inference_input,
            cls._get_generative_input,
            cls.inference,
            cls.generative,
            cls.loss,
            cls.sample,
            cls.marginal_ll,
        ]

    @classproperty
    def _metrics(cls):
        return ["validation_loss"]
