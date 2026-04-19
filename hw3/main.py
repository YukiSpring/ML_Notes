"""
date: 2026/04/19
id: 23375158
description:
    实现了基于PyTorch的LSTM模型用于多变量时序空气质量预测任务。
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

os.makedirs('asserts', exist_ok=True)


def prepare_train_data(file_path):
    df = pd.read_csv(file_path)

    def wind_encode(s):
        if s == "SE":
            return 1
        elif s == "NE":
            return 2
        elif s == "NW":
            return 3
        else:
            return 4

    df["wind_dir"] = df["wnd_dir"].apply(wind_encode)
    df.drop(['date', 'wnd_dir'], axis=1, inplace=True)
    df = df[['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain', 'wind_dir']]
    return df


def prepare_test_data(file_path):
    df = pd.read_csv(file_path)

    def wind_encode(s):
        if s == "SE":
            return 1
        elif s == "NE":
            return 2
        elif s == "NW":
            return 3
        else:
            return 4

    df["wind_dir"] = df["wnd_dir"].apply(wind_encode)
    df.drop(['wnd_dir'], axis=1, inplace=True)
    df = df[['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain', 'wind_dir']]
    return df


def series_to_supervised(data, n_in=1, n_out=1):
    df = pd.DataFrame(data)
    cols = []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    agg = pd.concat(cols, axis=1)
    agg.dropna(inplace=True)
    return agg


# Load data
train_df = prepare_train_data('hw3/LSTM-Multivariate_pollution.csv')
poll_mean = train_df['pollution'].mean()
poll_std = train_df['pollution'].std()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(train_df.values)

# Sliding window
reframed = series_to_supervised(scaled, 1, 1)
reframed = np.delete(reframed, [9, 10, 11, 12, 13, 14, 15], axis=1)

# Split data
n_train_hours = 24 * 365 * 3
train = reframed[:n_train_hours, :]
val = reframed[n_train_hours:, :]

X_train, y_train = train[:, :-1], train[:, -1]
X_val, y_val = val[:, :-1], val[:, -1]

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=72, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=72, shuffle=False)


# Model
class PollutionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(PollutionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


model = PollutionLSTM(input_size=8, hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 50
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_val_pred = model(X_val_t).numpy().squeeze()
    inv_y_pred = y_val_pred * poll_std + poll_mean
    inv_y_true = y_val * poll_std + poll_mean

rmse = np.sqrt(mean_squared_error(inv_y_true, inv_y_pred))
print(f'Validation RMSE = {rmse:.5f}')

# Plot
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('asserts/lstm_loss.png')
plt.close()

plt.figure(figsize=(14, 6))
plt.plot(inv_y_true[:200], label='Actual')
plt.plot(inv_y_pred[:200], label='Predicted')
plt.title(f'Validation RMSE = {rmse:.2f}')
plt.xlabel('Hours')
plt.ylabel('PM2.5')
plt.legend()
plt.savefig('asserts/lstm_prediction.png')
plt.close()

# Test set
test_df = prepare_test_data('hw3/pollution_test_data1.csv')
scaled_test = scaler.transform(test_df.values)
reframed_test = series_to_supervised(scaled_test, 1, 1)
reframed_test = np.delete(reframed_test, [9, 10, 11, 12, 13, 14, 15], axis=1)

X_test, y_test = reframed_test[:, :-1], reframed_test[:, -1]
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
X_test_t = torch.tensor(X_test, dtype=torch.float32)

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_t).numpy().squeeze()
    inv_y_test_pred = y_test_pred * poll_std + poll_mean
    inv_y_test_true = y_test * poll_std + poll_mean

test_rmse = np.sqrt(mean_squared_error(inv_y_test_true, inv_y_test_pred))
print(f'Test RMSE = {test_rmse:.5f}')

plt.figure(figsize=(14, 6))
plt.plot(inv_y_test_true, label='Actual')
plt.plot(inv_y_test_pred, label='Predicted')
plt.title(f'Test RMSE = {test_rmse:.2f}')
plt.xlabel('Hours')
plt.ylabel('PM2.5')
plt.legend()
plt.savefig('asserts/lstm_test_prediction.png')
plt.close()

# Full training model
X_all = reframed[:, :-1]
y_all = reframed[:, -1]
X_all = X_all.reshape((X_all.shape[0], 1, X_all.shape[1]))
X_all_t = torch.tensor(X_all, dtype=torch.float32)
y_all_t = torch.tensor(y_all, dtype=torch.float32)

model_final = PollutionLSTM(input_size=8, hidden_size=256)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_final.parameters(), lr=0.01)

for epoch in range(epochs):
    model_final.train()
    train_loss = 0
    for inputs, targets in DataLoader(TensorDataset(X_all_t, y_all_t), batch_size=256, shuffle=False):
        optimizer.zero_grad()
        outputs = model_final(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(DataLoader(TensorDataset(X_all_t, y_all_t), batch_size=256, shuffle=False))

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train: {train_loss:.6f}')

model_final.eval()
with torch.no_grad():
    y_final_pred = model_final(X_test_t).numpy().squeeze()
    inv_y_final_pred = y_final_pred * poll_std + poll_mean

final_rmse = np.sqrt(mean_squared_error(inv_y_test_true, inv_y_final_pred))
print(f'Final Test RMSE = {final_rmse:.5f}')

plt.figure(figsize=(14, 6))
plt.plot(inv_y_test_true, label='Actual')
plt.plot(inv_y_final_pred, label='Predicted')
plt.title(f'Final RMSE = {final_rmse:.2f}')
plt.xlabel('Hours')
plt.ylabel('PM2.5')
plt.legend()
plt.savefig('asserts/lstm_final_prediction.png')
plt.close()

print('Done!')