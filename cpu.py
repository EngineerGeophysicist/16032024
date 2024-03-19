import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

start_time = time.time()

# Выбор устройства: CPU или GPU
device = torch.device("cpu")
print(f"Устройство: {device}")

# Загрузка данных
df = pd.read_excel('df.xlsx')
df['Дата'] = pd.to_datetime(df['Дата'])
df.set_index('Дата', inplace=True)
data = df['Значение'].values.reshape(-1, 1)

# Масштабирование данных
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data)

# Создание последовательностей для обучения
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i+seq_length]
        sequences.append(sequence)
    return np.array(sequences)

seq_length = 12  # Длина последовательности, которую мы будем использовать для прогнозирования
X = create_sequences(data_normalized, seq_length)
y = data_normalized[seq_length:]

# Разделение на обучающую и тестовую выборки
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Преобразование в тензоры PyTorch
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Определение LSTM модели
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).to(device),
                            torch.zeros(1,1,self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Обучение LSTM модели
model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 150

for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq.to(device)) if torch.cuda.is_available() else model(seq)

        single_loss = loss_function(y_pred, labels.to(device)) if torch.cuda.is_available() else loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}, Device: {model.lstm.weight_ih_l0.device}')

# Прогнозирование на тестовой выборке
model.eval()
test_predictions = []

for i in range(len(X_test)):
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))
        test_predictions.append(model(X_test[i].to(device)) if torch.cuda.is_available() else model(X_test[i]))

# Обратное масштабирование данных
def inverse_normalize(data_normalized, data_min, data_max):
    return data_normalized * (data_max - data_min) + data_min

# Обратное масштабирование фактических значений тестовой выборки
y_test_unscaled = inverse_normalize(y_test.cpu().numpy(), df['Значение'].min(), df['Значение'].max())

# Обратное масштабирование прогнозируемых значений
test_predictions_unscaled = inverse_normalize(torch.tensor(test_predictions).cpu().numpy(), df['Значение'].min(), df['Значение'].max())

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.plot(df.index[train_size+seq_length:], y_test_unscaled, label='Фактические значения')
plt.plot(df.index[train_size+seq_length:], test_predictions_unscaled, label='Прогноз')
plt.title('Прогноз временного ряда с использованием LSTM')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.savefig('forecast_plot.png')  # Сохранение графика в файл
plt.show()

end_time = time.time()
execution_time = end_time - start_time
print("Время выполнения:", execution_time, "секунд")
