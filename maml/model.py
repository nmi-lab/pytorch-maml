import torch.nn as nn

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)


def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))

class MetaConvModel(MetaModule):
    """4-layer Convolutional Neural Network architecture from [1].

    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.

    out_features : int
        Number of classes (output of the model).

    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.

    feature_size : int (default: 64)
        Number of features returned by the convolutional head.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64, remove_time_dim = False):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.remove_time_dim = remove_time_dim

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))
        self.classifier = MetaLinear(feature_size, out_features, bias=True)

    def forward(self, inputs, params=None):
        if self.remove_time_dim: inputs = inputs[:,0]
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

class MetaMLPModel(MetaModule):
    """Multi-layer Perceptron architecture from [1].

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of classes (output of the model).

    hidden_sizes : list of int
        Size of the intermediate representations. The length of this list
        corresponds to the number of hidden layers.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
            MetaSequential(OrderedDict([
                ('linear', MetaLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('relu', nn.ReLU())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

def ModelConvOmniglot(out_features, hidden_size=64):
    return MetaConvModel(1, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size)

def ModelConvDoubleNMNIST(out_features, hidden_size=64):
    return MetaConvModel(2, out_features, hidden_size=hidden_size,
                         feature_size=256, remove_time_dim=True)

def ModelDECOLLE(out_features):
    from .meta_lenet_decolle import MetaLenetDECOLLE, DECOLLELoss, LIFLayerVariableTau, MetaLIFLayer, MetaLIFLayerNonorm, TimeWrappedMetaLenetDECOLLE
    from decolle.utils import parse_args, prepare_experiment, cross_entropy_one_hot
    import datetime, os, socket, tqdm
    import torch

    params_file = 'maml/decolle_params.yml'
    with open(params_file, 'r') as f:
        import yaml
        params = yaml.load(f)
    verbose = True

    #d, t = next(iter(gen_train))
    input_shape = params['input_shape']
    ## Create Model, Optimizer and Loss
    net = TimeWrappedMetaLenetDECOLLE(
                        out_channels=out_features,
                        Nhid=params['Nhid'],
                        Mhid=params['Mhid'],
                        kernel_size=params['kernel_size'],
                        pool_size=params['pool_size'],
                        stride = params['stride'],
                        input_shape=params['input_shape'],
                        alpha=params['alpha'],
                        alpharp=params['alpharp'],
                        beta=params['beta'],
                        num_conv_layers=params['num_conv_layers'],
                        num_mlp_layers=params['num_mlp_layers'],
                        lc_ampl=params['lc_ampl'],
                        lif_layer_type = MetaLIFLayerNonorm,
                        method=params['learning_method'],
                        with_output_layer=True).cuda()

    #Makes the network 16 bit.
    #net = net.half()


#     opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'], eps=1e-4)

#     if 'loss_scope' in params and params['loss_scope'] == 'bptt':
#         print('Using BPTT')
#         loss = [None for i in range(len(net))]
#         loss[-1] = torch.nn.SmoothL1Loss()

#         if net.with_output_layer:
#             loss[-1] = cross_entropy_one_hot
#             loss[-2] = torch.nn.MSELoss()
#         decolle_loss = DECOLLELoss(net = net, loss_fn = loss, reg_l=params['reg_l'])
#     else:
#         loss = [torch.nn.SmoothL1Loss() for i in range(len(net))]

#         if net.with_output_layer:
#             loss[-1] = cross_entropy_one_hot
#             loss[-2] = torch.nn.MSELoss()
#         decolle_loss = DECOLLELoss(net = net, loss_fn = loss, reg_l=params['reg_l'])

    ##Initialize
    net.init_parameters(torch.zeros([1,params['chunk_size_train']]+params['input_shape']).cuda())

    return net

def ModelConvMiniImagenet(out_features, hidden_size=64):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=5 * 5 * hidden_size)

def ModelMLPSinusoid(hidden_sizes=[40, 40]):
    return MetaMLPModel(1, 1, hidden_sizes)

if __name__ == '__main__':
    model = ModelDECOLLE(10)
