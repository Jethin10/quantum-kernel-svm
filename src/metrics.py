"""Kernel diagnostics, consistent PSD extension, and classifier statistics."""
import numpy as np
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.svm import SVC


def diagnostics(matrix, reference=None):
    matrix = np.asarray(matrix)
    sym = (matrix + matrix.T) / 2
    eigenvalues = np.linalg.eigvalsh(sym)
    result = {
        "symmetry_error": float(np.max(np.abs(matrix - matrix.T))),
        "diagonal_mean": float(np.diag(matrix).mean()),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "negative_eigenvalues": int(np.sum(eigenvalues < -1e-10)),
        "minimum_entry": float(matrix.min()), "maximum_entry": float(matrix.max()),
    }
    if reference is not None:
        result.update(
            kernel_mae=float(np.abs(matrix - reference).mean()),
            kernel_rmse=float(np.sqrt(np.mean((matrix - reference) ** 2))),
            relative_frobenius_error=float(np.linalg.norm(matrix - reference) /
                                           np.linalg.norm(reference)),
        )
    return result


def psd_extension(train, cross, threshold=1e-8):
    """Positive-eigenspace training features and consistent held-out extension.

    train features = V sqrt(L); test features = cross V / sqrt(L).
    Therefore repaired train = V L V.T, cross = cross V V.T.
    No test labels or test/test matrix participate. Keep raw inputs separately.
    """
    values, vectors = np.linalg.eigh((train + train.T) / 2)
    keep = values > threshold
    basis = vectors[:, keep]
    repaired = (basis * values[keep]) @ basis.T
    extended = cross @ basis @ basis.T
    return repaired, extended, {
        "retained_rank": int(keep.sum()), "threshold": threshold,
        "training_correction_frobenius": float(np.linalg.norm(repaired - train)),
        "cross_correction_frobenius": float(np.linalg.norm(extended - cross)),
    }


def fit_kernel(train, cross, ytrain, ytest, C, repair=False):
    t0 = time.perf_counter()
    correction = {}
    if repair:
        train, cross, correction = psd_extension(train, cross)
    preparation_seconds = time.perf_counter() - t0
    model = SVC(kernel="precomputed", C=C)
    t0 = time.perf_counter()
    model.fit(train, ytrain)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    prediction = model.predict(cross)
    prediction_seconds = time.perf_counter() - t0
    result = classification(ytest, prediction)
    result.update(support_vectors=int(len(model.support_)), C=float(C),
                  kernel_preparation_seconds=preparation_seconds,
                  svm_fit_seconds=fit_seconds, svm_prediction_seconds=prediction_seconds,
                  **correction)
    return model, result, prediction


def classification(ytrue, prediction):
    return {
        "accuracy": float(accuracy_score(ytrue, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(ytrue, prediction)),
        "f1": float(f1_score(ytrue, prediction)),
        "confusion_matrix": confusion_matrix(ytrue, prediction).tolist(),
    }


def readout_channel(p):
    one = np.array([[1 - p, p], [p, 1 - p]])
    channel = one
    for _ in range(3):
        channel = np.kron(channel, one)
    return channel


def wilson_interval(successes, n, z=1.959963984540054):
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [float(center - half), float(center + half)]
