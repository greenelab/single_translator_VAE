from scvi.data import synthetic_iid

from single_translator_VAE.vae import VAEModel


def test_VAEmodel():
    n_latent = 5
    adata = synthetic_iid()
    VAEModel.setup_anndata(adata, batch_key="batch", labels_key="labels", adata_obs="labels")

    print("Model Initiated..")
    model = VAEModel(adata, n_latent=n_latent, n_layers=10)
    model.train(max_epochs=5)
    model.get_elbo()
    model.get_latent_representation()
    model.get_marginal_ll(n_mc_samples=5)
    model.get_reconstruction_error()
    print("ELBO:", model.get_elbo())
    print("Latent Representation:", model.get_latent_representation)

    # tests __repr__
    print(model)
    print("\nSuccess!..")


test_VAEmodel()
