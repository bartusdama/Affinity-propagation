from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def calculate_similarity(X):
    n_samples = X.shape[0]
    similarity = -np.square(np.linalg.norm(X[:, None] - X[None, :], axis=2))

    preference = np.median(similarity)
    np.fill_diagonal(similarity, preference)

    return similarity


def update_responsibility_availability(S, R, A, damping):
    n_samples = S.shape[0]

    R_old = R.copy()

    AS = A + S

    max_AS = np.max(AS, axis=1)
    max_AS_k = AS - np.expand_dims(max_AS, axis=1)
    np.fill_diagonal(max_AS_k, 0)
    R = S - max_AS_k

    R = damping * R + (1 - damping) * R_old

    A_old = A.copy()

    Rp = np.maximum(0, R)
    Rp_diag = np.diag(R)
    A = np.minimum(0, np.sum(Rp, axis=0) - Rp)
    np.fill_diagonal(A, Rp_diag)

    A = damping * A + (1 - damping) * A_old

    return R, A


def assign_clusters(R, A):
    return np.argmax(A + R, axis=1)

def affinity_propagation(X, max_iter=200, convergence_iter=15, damping=0.9):
    n_samples = X.shape[0]

    S = calculate_similarity(X)

    R, A = np.zeros((n_samples, n_samples)), np.zeros((n_samples, n_samples))

    for iteration in range(max_iter):
        print(f"{iteration}/{max_iter} iteracji zrobionych")
        R, A = update_responsibility_availability(S, R, A, damping)

        if iteration > convergence_iter:
            clusters = assign_clusters(R, A)
            if np.all(clusters == assign_clusters(R, A)):
                print(f"Zbieżność osiągnięta po {iteration} iteracjach.")
                break

    clusters = assign_clusters(R, A)
    cluster_centers = np.unique(clusters)
    return clusters, cluster_centers


def load_mnist(sample_size=10000):
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target
    X = X[:sample_size]
    y = y[:sample_size].astype(int)
    print(X.shape)
    return X, y

def reduce_dimensions(X, n_components=50):
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

def print_centroids(centroids, pca, original_dim=28):
    centroids_original_space = pca.inverse_transform(centroids)
    fig, axes = plt.subplots(1, min(len(centroids), 10), figsize=(15,3))
    for i, ax in enumerate(axes):
        if i < len(centroids_original_space):
            ax.imshow(centroids_original_space[i].reshape(original_dim, original_dim), cmap='gray')
            ax.axis('off')
    plt.suptitle("Centroid klastrów")
    plt.show()


def main():
    X, y = load_mnist(sample_size=10000)
    print("Dane załadowane: ", X.shape)

    X_reduced, pca = reduce_dimensions(X)
    print("Dane po redukcji wymiarów: ", X_reduced.shape)

    clusters, centers = affinity_propagation(X_reduced)
    print(f"Liczba klastrów: {len(clusters)}")

    print_centroids(X_reduced[centers], pca)


#main()