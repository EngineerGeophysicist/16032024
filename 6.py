import pandas as pd
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

# Анализ стационарности
# Проверка стационарности с помощью графика автокорреляции
plot_acf(df['Значение'])
plt.show()

# Проверка стационарности с помощью графика частичной автокорреляции
plot_pacf(df['Значение'])
plt.show()

# Функция для выполнения теста Дики-Фуллера и вывода результатов
def adf_test(timeseries):
    result = adfuller(timeseries)
    print('Результаты теста Дики-Фуллера:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    # поскольку p-value < 0,05 мы отвергаем гипотезу о нестационарности данных
    print('Критические значения:')
    for key, value in result[4].items():
        print(f'  {key}: {value}')

# Выполнение теста Дики-Фуллера
adf_test(df['Значение'])

# Построение и обучение модели SARIMA
model = SARIMAX(df['Значение'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) # Параметры подлежат настройке
model_fit = model.fit()

# Прогнозирование на основе обученной модели
forecast = model_fit.forecast(steps=len(df))

# Визуализация фактических значений и прогноза
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Значение'], label='Фактические значения')
plt.plot(df.index, forecast, color='red', label='Прогноз')
plt.title('Прогноз временного ряда с использованием SARIMA')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.show()
