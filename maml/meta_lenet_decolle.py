#!/bin/python
#-----------------------------------------------------------------------------
# File Name : meta_lenet_decolle.py
# Author: Emre Neftci
#
# Creation Date : Tue 08 Sep 2020 11:18:03 AM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from decolle.base_model import *
from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)



class MetaLenetDECOLLE(DECOLLEBase,MetaModule):
    def __init__(self,
                 input_shape,
                 Nhid=[1],
                 Mhid=[128],
                 out_channels=1,
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 alpha=[.9],
                 beta=[.85],
                 alpharp=[.65],
                 dropout=[0.5],
                 num_conv_layers=2,
                 num_mlp_layers=1,
                 deltat=1000,
                 lc_ampl=.5,
                 lif_layer_type = LIFLayer,
                 method='rtrl',
                 with_output_layer = True):

        self.with_output_layer = with_output_layer
        if with_output_layer:
            Mhid += [out_channels]
            num_mlp_layers += 1
        self.num_layers = num_layers = num_conv_layers + num_mlp_layers
        # If only one value provided, then it is duplicated for each layer
        if len(kernel_size) == 1:   kernel_size = kernel_size * num_conv_layers
        if stride is None: stride=[1]
        if len(stride) == 1:        stride = stride * num_conv_layers
        if pool_size is None: pool_size = [1]
        if len(pool_size) == 1: pool_size = pool_size * num_conv_layers
        if len(alpha) == 1:         alpha = alpha * num_layers
        if len(alpharp) == 1:       alpharp = alpharp * num_layers
        if len(beta) == 1:          beta = beta * num_layers
        if len(dropout) == 1:       self.dropout = dropout = dropout * num_layers
        if Nhid is None:          self.Nhid = Nhid = []
        if Mhid is None:          self.Mhid = Mhid = []


        super(MetaLenetDECOLLE, self).__init__()

        # Computing padding to preserve feature size
        padding = (np.array(kernel_size) - 1) // 2  # TODO try to remove padding



        # THe following lists need to be nn.ModuleList in order for pytorch to properly load and save the state_dict
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_shape = input_shape
        Nhid = [input_shape[0]] + Nhid
        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers

        feature_height = self.input_shape[1]
        feature_width = self.input_shape[2]

        for i in range(self.num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width], 
                kernel_size = kernel_size[i],
                stride = stride[i],
                padding = padding[i],
                dilation = 1)
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            base_layer = MetaConv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i])
            layer = lif_layer_type(base_layer,
                             alpha=alpha[i],
                             beta=beta[i],
                             alpharp=alpharp[i],
                             deltat=deltat,
                             do_detach= True if method == 'rtrl' else False)
            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            readout = MetaLinear(int(feature_height * feature_width * Nhid[i + 1]), out_channels)

            # Readout layer has random fixed weights
            for param in readout.parameters():
                param.requires_grad = False
            self.reset_lc_parameters(readout, lc_ampl)

            dropout_layer = nn.Dropout(dropout[i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

        mlp_in = int(feature_height * feature_width * Nhid[-1])
        Mhid = [mlp_in] + Mhid
        for i in range(num_mlp_layers):
            base_layer = MetaLinear(Mhid[i], Mhid[i+1])
            layer = lif_layer_type(base_layer,
                             alpha=alpha[i],
                             beta=beta[i],
                             alpharp=alpharp[i],
                             deltat=deltat,
                             do_detach= True if method == 'rtrl' else False)
            
            if self.with_output_layer and i+1==num_mlp_layers:
                readout = nn.Identity()
                dropout_layer = nn.Identity()
            else:
                readout = MetaLinear(Mhid[i+1], out_channels)
                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False
                self.reset_lc_parameters(readout, lc_ampl)
                dropout_layer = nn.Dropout(dropout[self.num_conv_layers+i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

    def forward(self, input, params = None):
        s_out = []
        r_out = []
        u_out = []
        i = 0
        for lif, pool, ro, do in zip(self.LIF_layers, self.pool_layers, self.readout_layers, self.dropout_layers):
            if i == self.num_conv_layers: 
                input = input.view(input.size(0), -1)
            s, u = lif(input, self.get_subdict(params,'LIF_layers.{0}.base_layer'.format(i)))
            u_p = pool(u)
            if i+1 == self.num_layers:
                s_ = sigmoid(u_p)
            else:
                s_ = lif.sg_function(u_p)
            #sd_ = do(s_)
            #r_ = ro(sd_.reshape(sd_.size(0), -1))
            s_out.append(s_) 
            #r_out.append(r_)
            u_out.append(u_p)
            input = s_.detach() if lif.do_detach else s_
            i+=1

        return s_out, r_out, u_out

class MetaLIFLayer(MetaModule):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])
    sg_function = smooth_step

    def __init__(self, layer, alpha=.9, alpharp=.65, wrp=1.0, beta=.85, deltat=1000, do_detach=True):
        '''
        deltat: timestep in microseconds (not milliseconds!)
        '''
        super(MetaLIFLayer, self).__init__()
        self.base_layer = layer
        self.deltat = deltat
        self.dt = deltat/1e-6
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad=False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad=False)
        self.alpharp = alpharp
        self.wrp = wrp
        self.state = None
        self.do_detach = do_detach

    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.state = None
        self.base_layer = self.base_layer.cuda()
        return self

    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.state = None
        self.base_layer = self.base_layer.cpu()
        return self

    @staticmethod
    def reset_parameters(layer):
        return
        if has_attr(layer, 'out_channels'):
            conv_layer = layer
            n = conv_layer.in_channels
            for k in conv_layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            conv_layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if conv_layer.bias is not None:
                conv_layer.bias.data.uniform_(-stdv, stdv)
        elif hasattr(layer, 'out_features'): 
            layer.weight.data[:]*=0
            if layer.bias is not None:
                layer.bias.data.uniform_(-1e-3,1e-3)
        else:
            warnings.warn('Unhandled layer type, not resetting parameters')
    
    @staticmethod
    def get_out_channels(layer):
        '''
        Wrapper for returning number of output channels in a LIFLayer
        '''
        if hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'out_channels'): 
            return layer.out_channels
        elif hasattr(layer, 'get_out_channels'): 
            return layer.get_out_channels()
        else: 
            raise Exception('Unhandled base layer type')
    
    @staticmethod
    def get_out_shape(layer, input_shape):
        if hasattr(layer, 'out_channels'):
            return get_output_shape(input_shape, 
                                    kernel_size=layer.kernel_size,
                                    stride = layer.stride,
                                    padding = layer.padding,
                                    dilation = layer.dilation)
        elif hasattr(layer, 'out_features'): 
            return []
        elif hasattr(layer, 'get_out_shape'): 
            return layer.get_out_shape()
        else: 
            raise Exception('Unhandled base layer type')

    def init_state(self, Sin_t):
        dtype = Sin_t.dtype
        device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))

    def init_parameters(self, Sin_t):
        self.reset_parameters(self.base_layer)

    def forward(self, Sin_t, params=None):
        if self.state is None or (Sin_t.shape[0] != self.state.P.shape[0]):
            if Sin_t.shape[0] != self.state.P.shape[0]: print('readjusting batch size',Sin_t.shape[0], self.state.P.shape[0])
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + self.tau_s * Sin_t
        P = self.alpha * state.P + self.tau_m * state.Q  
        R = self.alpharp * state.R - state.S * self.wrp
        U = self.base_layer(P, params=params) + R
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U

    def get_output_shape(self, input_shape):
        layer = self.base_layer
        if hasattr(layer, 'out_channels'):
            im_height = input_shape[-2]
            im_width = input_shape[-1]
            height = int((im_height + 2 * layer.padding[0] - layer.dilation[0] *
                          (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1)
            weight = int((im_width + 2 * layer.padding[1] - layer.dilation[1] *
                          (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1)
            return [height, weight]
        else:
            return layer.out_features
    
    def get_device(self):
        return self.base_layer.weight.device
    
class MetaLIFLayerNonorm(MetaLIFLayer):
    sg_function  = sigmoid

    def forward(self, Sin_t, params = None ):
        if self.state is None:
            self.init_state(Sin_t)
        if Sin_t.shape[0] != self.state.P.shape[0]:
            #print('readjusting batch size',Sin_t.shape[0], self.state.P.shape[0])
            self.init_state(Sin_t)


        state = self.state
        Q = self.beta * state.Q + Sin_t
        P = self.alpha * state.P + state.Q  # TODO check with Emre: Q or state.Q?
        R = self.alpharp * state.R - state.S * self.wrp
        #Pc = (P>self.cutoff).type(P.dtype)/self.cutoff
        #Pd = (P-Pc).detach()+Pc
        U = self.base_layer(P, params=params) + R
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U
    
    def reset_parameters(self, layer):
        if hasattr(layer, 'out_channels'):
            conv_layer = layer
            n = conv_layer.in_channels
            for k in conv_layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250 * self.tau_s * self.tau_m
            conv_layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if conv_layer.bias is not None:
                conv_layer.bias.data.uniform_(-stdv, stdv)
        elif hasattr(layer, 'out_features'): 
            layer.weight.data[:]*=0
            if layer.bias is not None:
                layer.bias.data.uniform_(-1e-3,1e-3)
        else:
            warnings.warn('Unhandled data type, not resetting parameters')

class TimeWrappedMetaLenetDECOLLE(MetaLenetDECOLLE):
    def forward(self, Sin, params = None):
        self.init(Sin,burnin=200)
        t_sample = Sin.shape[1]
        out = torch.zeros([Sin.shape[0],self.LIF_layers[-1].base_layer.out_features]).to(Sin.device)
        for t in (range(200,t_sample)):
            Sin_t = Sin[:,t]
            out_ = super().forward(Sin_t, params=params)
            out+=out_[2][-1]
        return out/t_sample

    def forward_sequence(self, Sin, params = None, doinit=True):
        if doinit: 
            burnin=50
            self.init(Sin,burnin=burnin)
        t_sample = Sin.shape[1]
        out=[]
        for t in (range(0,t_sample)):
            Sin_t = Sin[:,t]
            out.append(super().forward(Sin_t, params=params))
        return out

    def init(self, data_batch, burnin):
        '''
        Necessary to reset the state of the network whenever a new batch is presented
        '''
        if self.requires_init is False:
            return
        for l in self.LIF_layers:
            l.state = None

        with torch.no_grad():
            self.forward_sequence(data_batch[:,:burnin],doinit=False)
        for l in self.LIF_layers:
            state_detach(l.state)

    def init_parameters(self, data_batch, params=None):
        Sin = data_batch[:, :, :, :]
        s_out = self.forward_sequence(Sin, params=params)[0][0]
        ins = [self.LIF_layers[0].state.Q]+s_out
        for i,l in enumerate(self.LIF_layers):
            l.init_parameters(ins[i])
    
    
    
if __name__ == "__main__":
    #Test building network
    net = MetaLenetDECOLLE(Nhid=[1,8],Mhid=[32,64],out_channels=10, input_shape=[1,28,28])
    d = torch.zeros([1,1,28,28])
    net(d)

