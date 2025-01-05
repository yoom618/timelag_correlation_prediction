import torch.nn as nn
from ._helpers import BaseModel, CNN1D_Base, MLP_Base



class CNNModel(BaseModel):
    ''' Initialize the model '''
    def __init__(self, input_size:tuple, output_size=1, 
                 cnn_hidden_list=[16,32], kernel_size=3, stride=1, padding=1, pool_size=2,
                 fc_hidden_list=[], dropout_p=0, batch_norm=False):
        '''
        input_size: 입력 데이터의 feature 수 ( input_size = (input_days, input_columns) )
        output_size: 출력 데이터의 차원 (default: 1)
        cnn_hidden_list: CNN 레이어의 hidden 차원 리스트 (default: [16,32])
        kernel_size: Conv1d 레이어의 kernel (default: 3)
        stride: Conv1d 레이어의 stride (default: 1)
        padding: Conv1d 레이어의 padding (default: 1)
        pool_size: MaxPool1d 레이어의 kernel 및 stride (default: 2)
        fc_hidden_list: FC layer의 hidden 차원 리스트 ([]일 경우, 1차원으로 요약하는 layer 하나만 적용)
        dropout_p: dropout 확률 (0~1, 0이면 dropout을 사용하지 않음)
        batch_norm: batch normalization 사용 여부
        '''
        super().__init__()

        self.cnn = CNN1D_Base(input_size=input_size, 
                              channel_list=cnn_hidden_list, kernel_size=kernel_size, stride=stride, padding=padding, pool_size=pool_size,
                              dropout=dropout_p, batchnorm=batch_norm)

        self.fc = MLP_Base(input_dim=self.cnn.output_dim * cnn_hidden_list[-1],
                           hidden_dims=fc_hidden_list, output_dim=output_size,
                           batchnorm=batch_norm, dropout=dropout_p)

        self.init_weights()

    
    def forward(self, x):
        # input x: (batch_size, input_days, input_columns)
        x = x.permute(0, 2, 1)      # (batch_size, input_columns, input_days)
        x = self.cnn(x)             # (batch_size, output_dim, output_len)
        x = x.view(x.size(0), -1)   # (batch_size, output_dim * output_len)
        x = self.fc(x)              # (batch_size, output_size)

        return x  # (batch_size, output_size)

    
    def conv(inp, oup, kernel, stride, pad, batch_norm, dropout_p):
            layers = []
            layers.append(nn.Conv1d(inp, oup, kernel, stride, pad, bias=True))
            if batch_norm:
                layers.append(nn.BatchNorm1d(oup))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p:
                layers.append(nn.Dropout(dropout_p))
            
            return nn.Sequential(*layers)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
