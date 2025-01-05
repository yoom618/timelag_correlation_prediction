import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 하이퍼 파라미터
input_size_ex = 24
input_size_comb = 24
hidden_size = 192
output_size = 1


# 데이터 생성 및 전처리 (슬라이딩 윈도우)
def create_sequences(data, input_size, output_size):
    sequences = []
    for i in range(len(data) - input_size - output_size + 1):
        # 입력 시퀀스 (input_size만큼 가져옴)
        seq_data = data[i:i + input_size]
        # 출력 시퀀스 (output_size만큼 가져옴)
        seq_target = data[i + input_size:i + input_size + output_size]
        sequences.append((seq_data, seq_target))
    return sequences
def create_combine_sequences(data, input_size, output_size):
    sequences = []
    for i in range(len(data) - input_size - output_size + 1):
        # 입력 시퀀스 (input_size만큼 가져옴)
        seq_data = data[i:i + input_size]
        # 출력 시퀀스 (output_size만큼 가져옴)
        seq_target = data[i + input_size:i + input_size + output_size]
        seq_target = seq_target[:,0]
        sequences.append((seq_data, seq_target))
    return sequences

# 데이터 로드 (가상의 데이터 예시)
df = pd.read_csv(os.path.join('prev_correlation', 'result.csv'))
# exchange: 환율 데이터, gole: 금값 데이터
exchange = df["X"].values  # 예제 환율 데이터
gold = df["Y"].values  # 예제 금값 데이터

# 라그 데이터 생성
lag = 12
gold_lagged = np.roll(gold, lag)
gold_lagged[:lag] = 0  # 초기 lag 값은 0으로 채움

# 데이터 결합
combined_data = np.vstack((exchange, gold_lagged)).T

# 정규화
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
exchange = scaler1.fit_transform(exchange.reshape(-1, 1)).flatten()
print(exchange.shape)
combined_data = scaler2.fit_transform(combined_data)
print(combined_data.shape)

# 데이터 슬라이딩 윈도우
exchange_sequences = create_sequences(exchange, input_size_ex, output_size)
combined_sequences = create_combine_sequences(combined_data, input_size_comb, output_size)

# 데이터 분리
train_ex, test_ex = exchange_sequences[:int(len(exchange_sequences)*0.8)],exchange_sequences[int(len(exchange_sequences)*0.8):]
train_comb, test_comb = combined_sequences[:int(len(combined_sequences)*0.8)],combined_sequences[int(len(combined_sequences)*0.8):]

# PyTorch 데이터셋
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

train_dataset_ex = TimeSeriesDataset(train_ex)
test_dataset_ex = TimeSeriesDataset(test_ex)
train_dataset_comb = TimeSeriesDataset(train_comb)
test_dataset_comb = TimeSeriesDataset(test_comb)
print(f'train_dataset_ex: {len(train_dataset_ex)} / {train_dataset_ex.__getitem__(0)[0].shape}, {train_dataset_ex.__getitem__(0)[1].shape}')
print(f'test_dataset_ex: {len(test_dataset_ex)} / {test_dataset_ex.__getitem__(0)[0].shape}, {test_dataset_ex.__getitem__(0)[1].shape}')
print(f'train_dataset_comb: {len(train_dataset_comb)} / {train_dataset_comb.__getitem__(0)[0].shape}, {train_dataset_comb.__getitem__(0)[1].shape}')
print(f'test_dataset_comb: {len(test_dataset_comb)} / {test_dataset_comb.__getitem__(0)[0].shape}, {test_dataset_comb.__getitem__(0)[1].shape}')
# pd.DataFrame([seq[0] for seq in train_dataset_ex.sequences]).to_csv('train_dataset_ex_X (minmax).csv', index=False)
# pd.DataFrame([seq[1] for seq in train_dataset_ex.sequences]).to_csv('train_dataset_ex_Y (minmax).csv', index=False)
# pd.DataFrame([seq[0] for seq in test_dataset_ex.sequences]).to_csv('test_dataset_ex_X (minmax).csv', index=False)
# pd.DataFrame([seq[1] for seq in test_dataset_ex.sequences]).to_csv('test_dataset_ex_Y (minmax).csv', index=False)
# pd.DataFrame([seq[0][:,0] for seq in train_dataset_comb.sequences]).to_csv('train_dataset_comb_X_exchange (minmax).csv', index=False)
# pd.DataFrame([seq[0][:,1] for seq in train_dataset_comb.sequences]).to_csv('train_dataset_comb_X_gold (minmax).csv', index=False)
# pd.DataFrame([seq[1] for seq in train_dataset_comb.sequences]).to_csv('train_dataset_comb_Y (minmax).csv', index=False)
# pd.DataFrame([seq[0][:,0] for seq in test_dataset_comb.sequences]).to_csv('test_dataset_comb_X_exchange (minmax).csv', index=False)
# pd.DataFrame([seq[0][:,1] for seq in test_dataset_comb.sequences]).to_csv('test_dataset_comb_X_gold (minmax).csv', index=False)
# pd.DataFrame([seq[1] for seq in test_dataset_comb.sequences]).to_csv('test_dataset_comb_Y (minmax).csv', index=False)


train_loader_ex = torch.utils.data.DataLoader(train_dataset_ex, batch_size=16, shuffle=True,drop_last=True)
test_loader_ex = torch.utils.data.DataLoader(test_dataset_ex, batch_size=16, shuffle=False,drop_last=True)
train_loader_comb = torch.utils.data.DataLoader(train_dataset_comb, batch_size=16, shuffle=True,drop_last=True)
test_loader_comb = torch.utils.data.DataLoader(test_dataset_comb, batch_size=16, shuffle=False,drop_last=True)

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 시점의 출력 사용
        return out

class LSTM_combind_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM_combind_Model, self).__init__()
        self.lstm = nn.LSTM(2, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size*2, output_size)
        self.fc = nn.Linear(hidden_size * input_size, output_size)

    def forward(self, x):
        # x = x.transpose(1,2)
        out, _ = self.lstm(x)
        # out = out.reshape(16, -1)  # [batch_size, num_sequences * hidden_size]
        out = out.reshape(out.size(0), -1)  # [batch_size, num_sequences * hidden_size]
        
        out = self.fc(out)  # 마지막 시점의 출력 사용
        
        return out

# 모델 학습 함수
def train_model(model, dataloader, optimizer, criterion, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 모델 평가 함수
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 모델 초기화 및 학습

# model_ex = LSTMModel(input_size_ex, hidden_size, output_size)
# model_comb = LSTM_combind_Model(input_size_comb, hidden_size, output_size)
model_ex = LSTMModel(1, hidden_size, output_size)
model_comb = LSTM_combind_Model(input_size_comb, hidden_size, output_size)
print(model_ex)
print(model_comb)

criterion = nn.MSELoss()
optimizer_ex = torch.optim.Adam(model_ex.parameters(), lr=0.001)
optimizer_comb = torch.optim.Adam(model_comb.parameters(), lr=0.001)

print("Training model with exchange data only...")
train_model(model_ex, train_loader_ex, optimizer_ex, criterion)
print("Training model with combined data...")
train_model(model_comb, train_loader_comb, optimizer_comb, criterion)

# 모델 평가
print("Evaluating model with exchange data only...")
loss_ex = evaluate_model(model_ex, test_loader_ex, criterion)
print("Evaluating model with combined data...")
loss_comb = evaluate_model(model_comb, test_loader_comb, criterion)

print(f"Loss (Exchange only): {loss_ex:.4f}")
print(f"Loss (Combined): {loss_comb:.4f}")



import matplotlib.pyplot as plt

# 예측 함수
def predict(model, dataloader):
    model.eval()
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)  # 예측값
            predictions.append(outputs.numpy())  # numpy로 변환하여 저장
            ground_truths.append(targets.numpy())  # 실제값 저장
    # 리스트를 numpy 배열로 변환
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    return predictions, ground_truths

# 모델별 예측값 계산
pred_ex, gt_ex = predict(model_ex, test_loader_ex)  # 모델1: exchange 데이터만 사용
pred_comb, gt_comb = predict(model_comb, test_loader_comb)  # 모델2: combined 데이터 사용

# Ground truth는 동일하므로 하나만 사용
gt = gt_ex

# Flatten (필요시)
pred_ex = pred_ex.flatten()
pred_comb = pred_comb.flatten()
gt = gt.flatten()





# 시각화
plt.figure(figsize=(12, 6))
plt.plot(gt, label="Ground Truth (Exchange)", color='black', linewidth=2)  # 실제값
plt.plot(pred_ex, label="Predictions (Exchange Only)", linestyle="--")  # 모델1 예측값
plt.plot(pred_comb, label="Predictions (Combined)", linestyle=":")  # 모델2 예측값
plt.title("Model Predictions vs Ground Truth")
plt.xlabel("Time Step")
plt.ylabel("Exchange Value")
plt.legend()
plt.grid()
plt.show()
