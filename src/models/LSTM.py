import torch.nn as nn
from ._helpers import BaseModel, MLP_Base


class LSTMModel(BaseModel):
    ''' Initialize the model '''
    def __init__(self, input_size:tuple, output_size=1, 
                 lstm_hidden_dim=64, lstm_n_layer=2, bidirectional=True, lstm_squeeze=False,
                 fc_hidden_list=[], dropout_p=0, batch_norm=False,):
        '''
        input_size: 입력 데이터의 feature 수 ( input_size = (input_days, input_columns) )
        output_size: 출력 데이터의 차원 (=1)
        lstm_hidden_dim: LSTM layer의 hidden 차원
        lstm_n_layer: LSTM layer의 layer 수
        fc_hidden_list: FC layer의 hidden 차원 리스트 ([]일 경우, 1차원으로 요약하는 layer 하나만 적용)
        dropout_p: dropout 확률 (0~1, 0이면 dropout을 사용하지 않음)
        batch_norm: batch normalization 사용 여부
        '''
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size[1], 
                            hidden_size=lstm_hidden_dim, 
                            num_layers=lstm_n_layer, 
                            batch_first=True, 
                            bidirectional=bidirectional, 
                            dropout=dropout_p)
        
        self.lstm_squeeze = lstm_squeeze
        input_dim = lstm_hidden_dim * (2 if bidirectional else 1) * input_size[0] if lstm_squeeze else lstm_hidden_dim * (2 if bidirectional else 1)
        self.fc = MLP_Base(input_dim=input_dim,
                           hidden_dims=fc_hidden_list, 
                           output_dim=output_size, 
                           batchnorm=batch_norm, 
                           dropout=dropout_p)

        self.init_weights()


    def forward(self, x):
        # input x: (batch_size, input_days, input_columns)
        x, _ = self.lstm(x)     # (batch_size, input_days, lstm_hidden_dim)
        if self.lstm_squeeze:   # 만일 lstm_squeeze가 True라면, 모든 hidden state를 사용
            x = x.reshape(x.size(0), -1) # (batch_size, input_days * lstm_hidden_dim)
        else:                   # 만일 lstm_squeeze가 False라면, 마지막 hidden state만 사용
            x = x[:, -1, :]     # (batch_size, lstm_hidden_dim)
        x = x.reshape(x.size(0), -1) # (batch_size, input_days * lstm_hidden_dim)
        x = self.fc(x)          # (batch_size, output_size)

        return x  # (batch_size, output_size)
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

