import numpy as np
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Conv2D, Dense


def spectral_edition(kernel, new_kernel):
    new_shape = new_kernel.shape

    u, s, vh = np.linalg.svd(kernel)
    new_u, new_s, new_vh = np.linalg.svd(new_kernel)

    # transfer s
    s_inter_size = min(len(s), len(new_s))
    # todo : this overwrite the largest sing values of new_s while keeping those might be interesting
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


def warmstart_Dense(from_layer: Dense, to_layer: Dense, include_bias=True):
    k1 = from_layer.kernel
    k2 = to_layer.kernel
    if include_bias:
        # todo: real method should be to find x such that kernel*x = bias and reduce it's dimension with the kernel SVD
        if from_layer.use_bias:
            k1 = np.hstack([k1, from_layer.bias])
        if to_layer.use_bias:
            k2 = np.hstack([k2, from_layer.bias])
    new_k2 = spectral_edition(k1, k2)
    if include_bias and to_layer.use_bias:
        to_layer.bias = new_k2[:, -1]
        new_k2 = new_k2[:, :-1]
    to_layer.kernel = new_k2


def warmstart_Conv2D(from_layer: Conv2D, to_layer: Conv2D):
    in_kernel_shape = from_layer.kernel.shape
    out_kernel_shape = to_layer.kernel.shape
    filter_blocks = []
    # 1. reshape all blocks
    for filter in range(in_kernel_shape[3]):
        channels_blocks = []
        for channel in range(in_kernel_shape[2]):
            block = from_layer.kernel[:, :, channel, filter]
            new_block = to_layer.kernel_initializer(to_layer.kernel_size, dtype=to_layer.dtype)
            # todo: handle biases ( same a as dense, should be refactored in spectral edition )
            new_block = spectral_edition(block, new_block)
            channels_blocks.append(new_block)
        if in_kernel_shape[2] > out_kernel_shape[2]:
            # reshape channels
            pass
        # append reshaped channels
        filter_blocks.append(channels_blocks)
    if in_kernel_shape[3] > out_kernel_shape[3]:
        # reshape filters
        pass
    # now we have the right amount of channels/filters put it in the new kernel


if __name__ == '__main__':
    pass
