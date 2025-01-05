import torch.nn as nn
from ._helpers import BaseModel, MLP_Base


class MLPModel(BaseModel):
    ''' Initialize the MLP model '''
    def __init__(self, input_size, n_hidden_list, output_size=1, dropout_p=0, batch_norm=False):
        '''
        n_hidden_list: 각 hidden layer의 노드 수를 담은 리스트
        dropout_p: dropout 확률 (0~1, 0이면 dropout을 사용하지 않음)
        batch_norm: batch normalization 사용 여부
        '''
        super().__init__()

        input_size = input_size[0] * input_size[1]   # input_days * input_columns
        
        self.mlp = MLP_Base(input_dim=input_size, hidden_dims=n_hidden_list, output_dim=output_size,
                            batchnorm=batch_norm, dropout=dropout_p)


    def forward(self, x):
        # input x: (batch_size, input_days, input_columns)
        x = x.view(x.size(0), -1)  # (batch_size, input_days * input_columns)
        x = self.mlp(x)            # (batch_size, output_size)

        return x                   # (batch_size, output_size)

