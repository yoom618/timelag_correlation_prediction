import numpy as np
import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self):
        """
        Forward pass logic
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    

class MLP_Base(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 batchnorm=True, dropout=0.2):
        super().__init__()

        # MLP layers (liner -> batchnorm -> relu -> dropout -> ... -> liner)
        self.layers = nn.Sequential()
        for idx, hidden_dim in enumerate(hidden_dims):
            self.layers.add_module(f'linear_{idx}', nn.Linear(input_dim, hidden_dim))
            if batchnorm:
                self.layers.add_module(f'bn_{idx}', nn.BatchNorm1d(hidden_dim))
            self.layers.add_module(f'act_{idx}', nn.ReLU())
            if dropout > 0:
                self.layers.add_module(f'dropout_{idx}', nn.Dropout(p=dropout))
            input_dim = hidden_dim
        self.layers.add_module(f'linear_{len(hidden_dims)}', nn.Linear(input_dim, output_dim))
        
        self._initialize_weights()


    def forward(self, x):
        return self.layers(x)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
    


class CNN1D_Base(nn.Module):
    def __init__(self, input_size: tuple,  # input_size = (seq_len, channel_size) 
                 channel_list=[8,16,32], kernel_size=3, stride=2, padding=1, pool_size=2,
                 dropout=0.2, batchnorm=True):
        super().__init__()

        # CNN 구조 : Conv1d -> BatchNorm1d -> ReLU -> Dropout 
        #           -> Conv1d -> BatchNorm1d -> ReLU -> Dropout -> MaxPool1d -> ...
        self.layers = nn.Sequential()
        cnn_output_len = input_size[0]
        in_channel_list = [input_size[1]] + channel_list[:-1]
        for idx, (in_channel, out_channel) in enumerate(zip(in_channel_list, channel_list)):
            self.layers.add_module(f'conv_{idx}', nn.Conv1d(in_channel, out_channel, 
                                                         kernel_size=kernel_size, 
                                                         stride=stride, padding=padding))
            cnn_output_len = (cnn_output_len + 2 * padding - kernel_size) // stride + 1  # (n + 2p - k) / s + 1
            if batchnorm:
                self.layers.add_module(f'bn_{idx}', nn.BatchNorm1d(out_channel))
            self.layers.add_module(f'act_{idx}', nn.ReLU())
            if dropout > 0:
                self.layers.add_module(f'dropout{idx}', nn.Dropout(p=dropout))
            if idx % 2 == 1:
                self.layers.add_module(f'pool_{idx}', nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
                cnn_output_len = (cnn_output_len - pool_size) // pool_size + 1  # (n - k) / s + 1

        self.output_dim = cnn_output_len

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                nn.init.constant_(m.bias.data, 0)
        

    def forward(self, x):
        # input x: (batch_size, seq_len, channel_size)
        x = self.layers(x)  # (batch_size, output_dim, output_len)

        return x