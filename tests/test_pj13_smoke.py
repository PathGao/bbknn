"""PJ13 smoke test: import, one minimal API call, shape assertions. Exact-NN (cKDTree) so no ANN randomness."""
import numpy as np

import bbknn
import bbknn.matrix


def _toy(n=60, d=10, seed=0):
    rng = np.random.RandomState(seed)
    pca = rng.normal(size=(n, d)).astype(np.float32)
    batch = np.array(["a", "b"] * (n // 2))
    return pca, batch


def test_import_version():
    assert isinstance(bbknn.__version__, str) and bbknn.__version__


def test_matrix_bbknn_shapes():
    pca, batch = _toy()
    n = pca.shape[0]
    distances, connectivities, params = bbknn.matrix.bbknn(
        pca, batch, neighbors_within_batch=3, n_pcs=pca.shape[1], computation="cKDTree"
    )
    assert distances.shape == (n, n)
    assert connectivities.shape == (n, n)
    assert connectivities.nnz > 0
    assert distances.getnnz(axis=1).max() <= 3 * 2
    assert isinstance(params, dict)


def test_anndata_bbknn():
    anndata = __import__("anndata")
    pca, batch = _toy()
    n = pca.shape[0]
    ad = anndata.AnnData(X=np.zeros((n, 5), dtype=np.float32))
    ad.obsm["X_pca"] = pca
    ad.obs["batch"] = batch
    bbknn.bbknn(ad, batch_key="batch", neighbors_within_batch=3, n_pcs=pca.shape[1], computation="cKDTree")
    assert ad.obsp["distances"].shape == (n, n)
    assert ad.obsp["connectivities"].shape == (n, n)
    assert "neighbors" in ad.uns
