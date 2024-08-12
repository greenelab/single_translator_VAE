from scvi.data import synthetic_iid

from single_translator_VAE.vae import VAEModel


def test_VAEmodel():
    n_latent = 5
    adata = synthetic_iid()
    adata = VAEModel.use_obs(adata, labels_key="labels_key", adata_obs=["batch", "labels"])
    adata_manager, adata = VAEModel.setup_anndata(adata, labels_key="labels_key", batch_key="batch")

    print("Model Initiated..")

    model = VAEModel(adata, n_latent=n_latent, n_layers=10)
    model.train(max_epochs=5)

    # tests __repr__
    print(model)
    print("\nSuccess!..")


adata = test_VAEmodel()
