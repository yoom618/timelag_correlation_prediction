# loss.py
# 예시로는 MSELoss, MAELoss, RMSELoss, MAPE가 정의되어 있음
# 각각의 클래스는 nn.Module을 상속받아 forward 메소드를 구현함
# 참고) torch.nn.Module 클래스 (https://pytorch.org/docs/stable/nn.html#loss-functions)

import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss as MAELoss

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss
    
class MAPE(nn.Module):
    def __init__(self):
        super(MAPE, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        loss = torch.mean(torch.abs((x - y) / y))
        return loss


if __name__ == '__main__':
    import torch
    
    output = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    target = torch.tensor([2, 2, 3, 4, 7], dtype=torch.float32)
    
    mse_err = MSELoss()
    rmse_err = RMSELoss()
    mae_err = MAELoss()
    mape_err = MAPE()

    print(f'MSE  : {mse_err(output, target):.4f}')          # 1.0000
    print(f'RMSE : {rmse_err(output, target):.4f}')         # 1.0000
    print(f'MAE  : {mae_err(output, target):.4f}')          # 0.6000
    print(f'MAPE : {mape_err(output, target) * 100:.2f} %') # 15.71 %