import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Загрузка данных из файла Excel
df = pd.read_excel('df.xlsx')

# Установка столбца "Дата" в качестве индекса и преобразование в тип данных datetime
df['Дата'] = pd.to_datetime(df['Дата'])
df.set_index('Дата', inplace=True)

# Визуализация временного ряда
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Значение'])
plt.title('Временной ряд')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.grid(True)
plt.show()

# Проверка на стационарность
def adf_test(timeseries):
    # Критерий Дики-Фуллера
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic: {:.3f}'.format(result[0]))
    print('p-value: {:.3f}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {:.3f}'.format(key, value))

adf_test(df['Значение'])

# Построение SARIMA модели
# Разделение данных на обучающий и тестовый наборы
train_data = df.iloc[:-12]  # Используем все данные, кроме последних 12 точек для обучения
test_data = df.iloc[-12:]   # Последние 12 точек для тестирования

# Выбор порядка SARIMA с помощью графиков автокорреляции и частичной автокорреляции
plot_acf(train_data, lags=20)
plot_pacf(train_data, lags=20)
plt.show()

# Прологарифмированный ряд
log_df = np.log(df['Значение'])

# Дифференцированный ряд
diff_df = df['Значение'].diff()

# Вычисление скользящего среднего
rolling_mean = df['Значение'].rolling(window=30).mean()

# Визуализация
plt.figure(figsize=(14, 10))

# График логарифмированного временного ряда
plt.subplot(3, 1, 1)
plt.plot(df.index, log_df, color='blue')
plt.title('Прологарифмированный временной ряд')
plt.xlabel('Дата')
plt.ylabel('Логарифм значения')
plt.grid(True)

# График дифференцированного временного ряда
plt.subplot(3, 1, 2)
plt.plot(df.index, diff_df, color='red')
plt.title('Дифференцированный временной ряд')
plt.xlabel('Дата')
plt.ylabel('Разность значений')
plt.grid(True)

# График скользящего среднего
plt.subplot(3, 1, 3)
plt.plot(df.index, rolling_mean, label='Средние значения', color='green')
plt.title('Средние значения временного ряда')
plt.xlabel('Дата')
plt.ylabel('Среднее значение')
plt.grid(True)

plt.tight_layout()
plt.show()

# Подбор параметров SARIMA
model = SARIMAX(train_data['Значение'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result = model.fit()
print(result.summary())

# Прогнозирование тренда
forecast = result.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

# Визуализация прогноза
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['Значение'], label='Обучающий набор')
plt.plot(test_data.index, test_data['Значение'], label='Тестовый набор')
plt.plot(test_data.index, forecast, label='Прогноз')
plt.title('Прогноз SARIMA модели')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.show()
