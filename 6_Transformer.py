import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

# Устройство: GPU или CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

seq_length = 12
X = create_sequences(data_normalized, seq_length)
y = data_normalized[seq_length:]

# Разделение на обучающую и тестовую выборки
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Преобразование в тензоры PyTorch
X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# Определение модели Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, output_size=1, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # Перестановка размерностей
        output = self.transformer_encoder(src)
        output = self.linear(output[-1])  # Используем только последний выход
        return output

# Обучение модели Transformer
model = TransformerModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 150

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 25 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# Прогнозирование на тестовой выборке
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# Обратное масштабирование данных
def inverse_normalize(data_normalized, data_min, data_max):
    return data_normalized * (data_max - data_min) + data_min

y_test_unscaled = inverse_normalize(y_test.cpu().numpy(), df['Значение'].min(), df['Значение'].max())
y_pred_unscaled = inverse_normalize(y_pred.cpu().numpy(), df['Значение'].min(), df['Значение'].max())

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.plot(df.index[train_size+seq_length:], y_test_unscaled, label='Фактические значения')
plt.plot(df.index[train_size+seq_length:], y_pred_unscaled, label='Прогноз')
plt.title('Прогноз временного ряда с использованием модели Transformer')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.savefig('forecast_plot_transformer.png')  # Сохранение графика в файл
plt.show()

end_time = time.time()
execution_time = end_time - start_time
print("Время выполнения:", execution_time, "секунд")
#это говно не работает
