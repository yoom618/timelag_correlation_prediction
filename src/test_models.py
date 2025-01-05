from models import *


if __name__ == '__main__':
    
    import torch

    # 10일 동안의 3개 feature를 가진 데이터로 1개의 output을 예측하는 모델
    batch_size = 7
    input_days = 10
    input_dim = 3

    # model = MLP(input_size=(input_days, input_dim),
    #             n_hidden_list=[512, 512, 512], 
    #             dropout_p=0.2, 
    #             batch_norm=True)
    
    # model = CNN(input_size=(input_days, input_dim),
    #             cnn_hidden_list=[32, 64],
    #             fc_hidden_list=[64, 32],
    #             dropout_p=0.2,
    #             batch_norm=True)

    model = LSTM(input_size=(input_days, input_dim),
                 lstm_hidden_dim=16,
                 lstm_n_layer=2,
                 bidirectional=True,
                 fc_hidden_list=[128, 64],
                 dropout_p=0.2, 
                 batch_norm=True)

    print(model)

    x = torch.randn(batch_size, input_days, input_dim)
    if isinstance(model, MLP):
        x = x.view(x.size(0), -1)  # (batch_size, input_days * input_dim)
    y = model(x)
    print(f'\nx: {x.shape} => y: {y.shape}')