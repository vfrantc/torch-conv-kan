import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState
from scipy.stats import chi

def check_input(input):
    if input.dim() not in {2, 3, 4, 5}:
        raise RuntimeError(
            "Quaternion linear accepts only input of dimension 2 or 3. Quaternion conv accepts up to 5 dim "
            " input.dim = " + str(input.dim())
        )

    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size(1)

    if nb_hidden % 4 != 0:
        raise RuntimeError(
            "Quaternion Tensors must be divisible by 4."
            " input.size()[1] = " + str(nb_hidden)
        )

def get_r(input):
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size(-1)
    else:
        nb_hidden = input.size(1)

    if input.dim() == 2:
        return input.narrow(1, 0, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, 0, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, 0, nb_hidden // 4)

def get_i(input):
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size(-1)
    else:
        nb_hidden = input.size(1)
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 4, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden // 4, nb_hidden // 4)

def get_j(input):
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size(-1)
    else:
        nb_hidden = input.size(1)
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 2, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 2, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden // 2, nb_hidden // 4)

def get_k(input):
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size(-1)
    else:
        nb_hidden = input.size(1)
    if input.dim() == 2:
        return input.narrow(1, nb_hidden - nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden - nb_hidden // 4, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden - nb_hidden // 4, nb_hidden // 4)

def quaternion_conv(input, r_weight, i_weight, j_weight, k_weight, bias, stride,
                    padding, groups, dilation):
    """
    Applies a quaternion convolution to the incoming data:
    """

    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
    cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=1)
    cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=1)
    cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=1)

    cat_kernels_4_quaternion = torch.cat(
        [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0
    )

    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception(
            "The convolutional input is either 3, 4 or 5 dimensions."
            " input.dim = " + str(input.dim())
        )

    return convfunc(input, cat_kernels_4_quaternion, bias, stride, padding, dilation, groups)

def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if criterion == 'glorot':
        s = 1.0 / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        s = 1.0 / np.sqrt(2 * fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState(np.random.randint(1, 1234))

    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if isinstance(kernel_size, int):
            kernel_shape = (out_features, in_features) + (kernel_size,)
        else:
            kernel_shape = (out_features, in_features) + tuple(kernel_size)

    modulus = chi.rvs(4, loc=0, scale=s, size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)

    # Purely imaginary quaternions unitary
    for i in range(number_of_weights):
        norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i * np.sin(phase)
    weight_j = modulus * v_j * np.sin(phase)
    weight_k = modulus * v_k * np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)

def affect_init_conv(r_weight, i_weight, j_weight, k_weight, kernel_size, init_func, rng,
                     init_criterion):
    if (
        r_weight.size() != i_weight.size()
        or r_weight.size() != j_weight.size()
        or r_weight.size() != k_weight.size()
    ):
        raise ValueError(
            'The real and imaginary weights '
            'should have the same size . Found: r:'
            + str(r_weight.size())
            + ' i:'
            + str(i_weight.size())
            + ' j:'
            + str(j_weight.size())
            + ' k:'
            + str(k_weight.size())
        )

    elif 2 >= r_weight.dim():
        raise Exception(
            'affect_conv_init accepts only tensors that have more than 2 dimensions. Found dimension = '
            + str(r_weight.dim())
        )

    r, i, j, k = init_func(
        r_weight.size(1),
        r_weight.size(0),
        rng=rng,
        kernel_size=kernel_size,
        criterion=init_criterion,
    )
    r, i, j, k = (
        torch.from_numpy(r),
        torch.from_numpy(i),
        torch.from_numpy(j),
        torch.from_numpy(k),
    )
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)

def get_kernel_and_weight_shape(operation, in_channels, out_channels, kernel_size):
    if operation == 'convolution1d':
        if not isinstance(kernel_size, int):
            raise ValueError(
                "An invalid kernel_size was supplied for a 1d convolution. The kernel size"
                " must be integer in this case. Found kernel_size = " + str(kernel_size)
            )
        else:
            ks = kernel_size
            w_shape = (out_channels, in_channels) + (ks,)
    else:  # in case it is 2d or 3d.
        if operation == 'convolution2d' and isinstance(kernel_size, int):
            ks = (kernel_size, kernel_size)
        elif operation == 'convolution3d' and isinstance(kernel_size, int):
            ks = (kernel_size, kernel_size, kernel_size)
        elif not isinstance(kernel_size, int):
            if operation == 'convolution2d' and len(kernel_size) != 2:
                raise ValueError(
                    "An invalid kernel_size was supplied for a 2d convolution. The kernel size"
                    " must be either an integer or a tuple of 2. Found kernel_size = "
                    + str(kernel_size)
                )
            elif operation == 'convolution3d' and len(kernel_size) != 3:
                raise ValueError(
                    "An invalid kernel_size was supplied for a 3d convolution. The kernel size"
                    " must be either an integer or a tuple of 3. Found kernel_size = "
                    + str(kernel_size)
                )
            else:
                ks = kernel_size
        w_shape = (out_channels, in_channels) + ks
    return ks, w_shape

# QConv2d class

class QConv2d(nn.Module):
    """
    Quaternion Convolution 2D layer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        init_criterion='glorot',
        weight_init='quaternion',
        seed=None,
        rotation=False,
        quaternion_format=True,
        scale=False,
    ):
        super(QConv2d, self).__init__()

        if in_channels % 4 != 0 or out_channels % 4 != 0:
            raise ValueError(
                "in_channels and out_channels must be divisible by 4 for quaternion convolution"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.quaternion_in_channels = in_channels // 4
        self.quaternion_out_channels = out_channels // 4

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias
        self.padding_mode = padding_mode

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.scale = scale

        # Set operation to 'convolution2d'
        self.operation = 'convolution2d'
        self.winit = {
            'quaternion': quaternion_init,
            'unitary': quaternion_init,  # Assuming unitary_init is similar
            'random': quaternion_init,   # Assuming random_init is similar
        }[self.weight_init]

        # Compute kernel size and weight shape
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        _, self.w_shape = get_kernel_and_weight_shape(
            self.operation,
            self.quaternion_in_channels,
            self.quaternion_out_channels,
            self.kernel_size,
        )

        self.r_weight = nn.Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = nn.Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = nn.Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = nn.Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param = nn.Parameter(torch.Tensor(*self.w_shape))
        else:
            self.scale_param = None

        if self.rotation:
            self.zero_kernel = nn.Parameter(
                torch.zeros(self.r_weight.shape), requires_grad=False
            )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights
        affect_init_conv(
            self.r_weight,
            self.i_weight,
            self.j_weight,
            self.k_weight,
            self.kernel_size,
            self.winit,
            self.rng,
            self.init_criterion,
        )
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.rotation:
            return quaternion_conv_rotation(
                input,
                self.zero_kernel,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.bias,
                self.stride,
                self.padding,
                self.groups,
                self.dilation,
                self.quaternion_format,
                self.scale_param,
            )
        else:
            return quaternion_conv(
                input,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.bias,
                self.stride,
                self.padding,
                self.groups,
                self.dilation,
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + '('
            + 'in_channels='
            + str(self.in_channels)
            + ', out_channels='
            + str(self.out_channels)
            + ', bias='
            + str(self.bias_flag)
            + ', kernel_size='
            + str(self.kernel_size)
            + ', stride='
            + str(self.stride)
            + ', padding='
            + str(self.padding)
            + ', dilation='
            + str(self.dilation)
            + ', init_criterion='
            + str(self.init_criterion)
            + ', weight_init='
            + str(self.weight_init)
            + ', seed='
            + str(self.seed)
            + ', rotation='
            + str(self.rotation)
            + ', q_format='
            + str(self.quaternion_format)
            + ')'
        )


class QKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, spline_order, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0,
                 **norm_kwargs):
        super(QKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        self.spline_conv = nn.ModuleList([conv_class((grid_size + spline_order) * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1,
                                                     bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        )
        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_kan(self, x, group_index):

        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))

        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
        # Compute the basis for the spline using intervals and input values.
        target = x.shape[1:] + self.grid.shape
        grid = self.grid.view(*list([1 for _ in range(self.ndim + 1)] + [-1, ])).expand(target).contiguous().to(
            x.device)

        bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[..., :-(k + 1)]
            right_intervals = grid[..., k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + \
                    ((grid[..., k + 1:] - x_uns) / (grid[..., k + 1:] - grid[..., 1:(-k)]) * bases[..., 1:])
        bases = bases.contiguous()
        bases = bases.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](bases)
        x = self.prelus[group_index](self.layer_norm[group_index](base_output + spline_output))

        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y



class QKANConv2DLayer(QKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm2d,
                 **norm_kwargs):
        super(QKANConv2DLayer, self).__init__(QConv2d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=2,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)

