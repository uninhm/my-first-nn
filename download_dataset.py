from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data.astype(np.uint8)      # shape (70000, 784)
y = mnist.target.astype(np.uint8)    # shape (70000,)

# Reshape the images just in case
X = X.reshape(-1, 28*28)

np.savez_compressed(
    "mnist.npz",
    X=X,
    y=y
)

