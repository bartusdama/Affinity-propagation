import numpy as np
from sklearn.datasets import fetch_openml

def calculate_similarity(X):
    S = -np.linalg.norm(X[:, np.newaxis] - X, axis=2)
    return S

def update_responsibility(S, A, R):
    AS = A + S
    max_vals = np.max(AS, axis=1, keepdims=True)
    R[:] = S - max_vals
    np.fill_diagonal(R, S.diagonal() - np.partition(AS, -2, axis=1)[:, -2])


def update_availability(R, A):
    Rp = np.maximum(R, 0)
    np.fill_diagonal(Rp, R.diagonal())
    A[:] = np.minimum(0, Rp.sum(axis=0, keepdims=True) - Rp)
    np.fill_diagonal(A, Rp.sum(axis=0) - Rp.diagonal())

def affinity_propagation(X, max_iter=100, damping=0.5):
    S = calculate_similarity(X)
    n = S.shape[0]

    R = np.zeros((n, n))
    A = np.zeros((n, n))

    for it in range(max_iter):
        R_old = R.copy()
        A_old = A.copy()

        update_responsibility(S, A, R)
        update_availability(R, A)

        R = damping * R_old + (1 - damping) * R
        A = damping * A_old + (1 - damping) * R

    exampler_indicates = np.argmax(A + R, axis=1)
    return exampler_indicates

if __name__ == '__main__':
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data[:50]

    exemplars = affinity_propagation(X.to_numpy(), max_iter=100, damping=0.5)

    unique_exemplars = np.unique(exemplars)
    print(f"Liczba centrów klastrów: {len(unique_exemplars)}")
    print(f"Centra klastrów (indeksy): {unique_exemplars}")