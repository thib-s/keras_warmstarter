from unittest import TestCase
import numpy as np
from warmstarter.strategies import spectral_edition


class Test(TestCase):

    def test_spectral_edition_identity(self):
        init_shape = (4, 3)
        for i in range(10):
            kernel = np.random.sample(init_shape)
            self.assertTrue(np.allclose(kernel, spectral_edition(kernel, init_shape)),
                            "passing original shape must not change the kernel")
            init_shape = (8, 10)
            kernel = np.random.sample(init_shape)
            self.assertTrue(np.allclose(kernel, spectral_edition(kernel, init_shape)),
                            "passing original shape must not change the kernel")

    def test_spectral_edition_conservation(self):
        init_shape = (4, 3)
        for i in range(10):
            kernel = np.random.sample(init_shape)
            large_kernel = spectral_edition(kernel, (4, 5))
            kernel_bar = spectral_edition(large_kernel, init_shape)
            self.assertTrue(np.allclose(kernel, kernel_bar),
                            "reshaping to large kernel and reshaping back to initial shape must not change kernel")

            large_kernel = spectral_edition(kernel, (7, 3))
            kernel_bar = spectral_edition(large_kernel, init_shape)
            self.assertTrue(np.allclose(kernel, kernel_bar),
                            "reshaping to large kernel and reshaping back to initial shape must not change kernel")

            large_kernel = spectral_edition(kernel, (7, 5))
            kernel_bar = spectral_edition(large_kernel, init_shape)
            self.assertTrue(np.allclose(kernel, kernel_bar),
                            "reshaping to large kernel and reshaping back to initial shape must not change kernel")

    def test_spectral_edition_composition(self):
        init_shape = (4, 3)
        final_shape = 10, 9
        for i in range(10):
            kernel = np.random.sample(init_shape)
            larger_kernel_direct = spectral_edition(kernel, final_shape)

            large_kernel = spectral_edition(kernel, (final_shape[0], 5))
            larger_kernel = spectral_edition(large_kernel, final_shape)
            self.assertTrue(np.allclose(larger_kernel_direct, larger_kernel),
                            "reshaping in two steps must be equivalent to reshaping in single step")

            large_kernel = spectral_edition(kernel, (8, final_shape[1]))
            larger_kernel = spectral_edition(large_kernel, final_shape)
            self.assertTrue(np.allclose(larger_kernel_direct, larger_kernel),
                            "reshaping in two steps must be equivalent to reshaping in single step")

            large_kernel = spectral_edition(kernel, (8, 5))
            larger_kernel = spectral_edition(large_kernel, final_shape)
            self.assertTrue(np.allclose(larger_kernel_direct, larger_kernel),
                            "reshaping in two steps must be equivalent to reshaping in single step")