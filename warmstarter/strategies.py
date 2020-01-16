import numpy as np
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Conv2D


def spectral_edition(kernel, new_shape:tuple, kernel_initializer: Initializer=None):
    # initialize new kernel
    if kernel_initializer is None:
        new_u = np.zeros((new_shape[0], new_shape[0]))
        new_s = np.zeros(min(new_shape))
        new_vh = np.zeros((new_shape[1], new_shape[1]))
    else:
        new_k = kernel_initializer(shape=new_shape)
        new_u, new_s, new_vh = np.linalg.svd(new_k)

    u, s, vh = np.linalg.svd(kernel)

    # transfer s
    s_inter_size = min(len(s), len(new_s))
    new_s[:s_inter_size] = s[:s_inter_size]
    inflated_s = np.zeros(new_shape)
    np.fill_diagonal(inflated_s, new_s)

    # transfer vh
    vh_inter_size = min(new_shape[1], vh.shape[1])
    new_vh[:vh_inter_size, :vh_inter_size] = vh[:vh_inter_size, :vh_inter_size]

    # transfer u
    u_inter_size = min(new_shape[0], u.shape[0])
    new_u[:u_inter_size, :u_inter_size] = u[:u_inter_size, :u_inter_size]

    # combine all to get kernel
    return np.matmul(new_u, np.matmul(inflated_s, new_vh))


def warmstart_Conv2D(from_layer: Conv2D, to_layer: Conv2D):
    in_kernel_shape = from_layer.kernel.shape
    for channel in range(in_kernel_shape[2]):
        for filter in range(in_kernel_shape[3]):
            block = from_layer.kernel[:, :, channel, filter]


if __name__ == '__main__':
    kernel = np.random.sample((4, 3))
    k2 = spectral_edition(kernel, (4, 3))  # no changes
    k3 = spectral_edition(kernel, (4, 4))  # one extra input
    k4 = spectral_edition(kernel, (4, 2))  # one input removed

    k5 = spectral_edition(kernel, (5, 3))  # one extra output
    k6 = spectral_edition(kernel, (2, 3))  # one output removed

    k7 = spectral_edition(kernel, (5, 4))  # extra input & extra output
    k8 = spectral_edition(kernel, (3, 2))  # fewer input & fewer output
    k9 = spectral_edition(kernel, (5, 2))  # fewer input & extra output
    k10 = spectral_edition(kernel, (5, 2))  # extra input & fewer output
    pass
