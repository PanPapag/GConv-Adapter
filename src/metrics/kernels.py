"""
Code taken and slightly modified from: https://github.com/BorgwardtLab/ggme/blob/main/src/metrics/utils.py

Kernels for comparing graph summary statistics or representations.
"""

from functools import partial
from typing import Callable, Union, Optional, List
from sklearn.base import TransformerMixin
from sklearn.metrics import pairwise_kernels
from sklearn.gaussian_process.kernels import Kernel
from scipy.linalg import toeplitz
import pyemd
import numpy as np

from src.utils.metrics import ensure_padded


def laplacian_total_variation_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Geodesic Laplacian kernel based on total variation distance.

    Args:
        x (np.ndarray): First input vector.
        y (np.ndarray): Second input vector.
        sigma (float, optional): Kernel parameter. Default is 1.0.

    Returns:
        float: Kernel value based on total variation distance.
    """
    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-sigma * dist)


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Gaussian (RBF) kernel.

    Args:
        x (np.ndarray): First input vector.
        y (np.ndarray): Second input vector.
        sigma (float, optional): Kernel parameter. Default is 1.0.

    Returns:
        float: Kernel value based on Gaussian (RBF) kernel.
    """
    return np.exp(-sigma * np.dot(x - y, x - y))


def gaussian_tv(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Gaussian kernel based on total variation distance.

    Args:
        x (np.ndarray): First input vector.
        y (np.ndarray): Second input vector.
        sigma (float, optional): Kernel parameter. Default is 1.0.

    Returns:
        float: Kernel value based on total variation distance.

    (Note): Implementation taken from GRAN code
    """
    support_size = max(len(x), len(y))
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_emd(x: np.ndarray, y: np.ndarray, sigma: float = 1.0, distance_scaling: float = 1.0) -> float:
    """
    Gaussian kernel with squared distance in exponential term replaced by EMD.

    Args:
        x (np.ndarray): First input vector.
        y (np.ndarray): Second input vector.
        sigma (float, optional): Standard deviation. Default is 1.0.
        distance_scaling (float, optional): Scaling factor for distance. Default is 1.0.

    Returns:
        float: Kernel value based on Earth Mover's Distance (EMD).

    (Note): Implementation taken from GraphRNN code
    """
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))


def linear_kernel(x: np.ndarray, y: np.ndarray, normalize: bool = False) -> float:
    """
    Linear kernel.

    Args:
        x (np.ndarray): First input vector.
        y (np.ndarray): Second input vector.
        normalize (bool, optional): If True, the kernel value is normalized. Default is False.

    Returns:
        float: Kernel value based on linear kernel.
    """
    if normalize:
        return np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
    return np.dot(x, y)


class KernelDistributionWrapper:
    """
    Wrap kernel function to work with distributions.

    The purpose of this class is to wrap a kernel function or other class
    such that it can be directly used with distributions of samples.
    """

    def __init__(self, kernel: Callable, pad: bool = True, **kwargs):
        """
        Create new wrapped kernel.

        Args:
            kernel (callable): Kernel function to use for the calculation of a kernel value
                between two samples from a distribution. The wrapper ensures
                that the kernel can be evaluated for distributions, not only
                for samples.
            pad (bool, optional): If set, ensures that all input sequences will be of the same
                length. Default is True.
            **kwargs: Additional arguments for the kernel function.
        """
        self.pad = pad
        self.kernel_type = "kernel_fn"

        # Check whether we have something more complicated than
        # a regular kernel function.
        if isinstance(kernel, TransformerMixin):
            self.kernel_type = "transformer"
        elif isinstance(kernel, Kernel):
            self.kernel_type = "passthrough"

        self.original_kernel = partial(kernel, **kwargs)

    def __call__(self, X: Union[np.ndarray, List], Y: Optional[Union[np.ndarray, List]] = None) -> np.ndarray:
        """
        Call kernel wrapper for two arguments.
        The specifics of this evaluation depend on the kernel that is
        wrapped; the function can call the proper method of a kernel,
        thus ensuring that the output array is always of shape (n, m)
        with n and m being the lengths of the input distributions.

        Args:
            X (array-like): First input distribution. Needs to be compatible with the
                wrapped kernel function.
            Y (array-like, optional): Second input distribution. The same caveats as for the
                first one apply. Default is None.

        Returns:
            np.ndarray: Kernel matrix between the samples of X and Y.
        """
        if self.kernel_type == "transformer":
            return self.original_kernel.transform(X, Y)
        elif self.kernel_type == "kernel_fn":

            if self.pad:
                X, Y = ensure_padded(X, Y)

            return pairwise_kernels(X, Y, metric=self.original_kernel)

        # By default: just evaluate the kernel!
        return self.original_kernel(X, Y)

    def diag(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> List[float]:
        """
        Return diagonal values of the wrapped kernel.

        Args:
            X (array-like): First input distribution.
            Y (array-like, optional): Second input distribution. Default is None.

        Returns:
            list: Diagonal values of the kernel matrix.
        """
        if Y is None:
            Y = X

        if self.kernel_type == "transformer":
            raise NotImplementedError()
        elif self.kernel_type == "kernel_fn":

            if self.pad:
                X, Y = ensure_padded(X, Y)

            return [self.original_kernel(x, y) for x, y in zip(X, Y)]

        return self.original_kernel.diag(X, Y)
