from torch import nn
from torch.nn.functional import grid_sample
from dpipe.layers import Reshape
from dpipe.torch.utils import to_var, is_on_cuda
import numpy as np


class DefConv(nn.Conv2d):
    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        super(DefConv, self).__init__(filters, filters * 2, 3, padding=1, bias=False, **kwargs)

        self.filters = filters

        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

        self.reshape_bc_offset = Reshape(-1, '2', '3', 2)
        self.reshape_bc_x = Reshape(-1, '2', '3')

        self.reshape_back = Reshape(-1, self.filters, '1', '2')

    def identity_mapping(self, shape, normalize=True):
        grid = np.mgrid[tuple(map(slice, shape))].astype('float32')

        if normalize:
            grid = grid / (np.array(shape) - 1).reshape((-1,) + (1,) * len(shape))
            grid = 2 * grid - 1

        return to_var(grid, is_on_cuda(self)).float()

    def _get_grid(self, shape):

        if not hasattr(self, 'grid'):
            self.grid = self.identity_mapping(shape)

        return self.grid

    def _bilinear_grid_sample(self, image, grid):
        grid = grid.flip(-1)
        return grid_sample(image, grid)

    def forward(self, x):
        x_shape = x.size()
        offsets = super(DefConv, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)
        offsets = self.reshape_bc_offset(offsets)

        # x: (b*c, h, w)
        x = self.reshape_bc_x(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = self._bilinear_grid_sample(x, self._get_grid(self, x_shape[2:]) + offsets)

        # x_offset: (b, h, w, c)
        x_offset = self.reshape_back(x_offset)

        return x_offset
