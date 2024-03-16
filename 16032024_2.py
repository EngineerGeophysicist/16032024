import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
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

# Построение SARIMA модели с использованием автоматического выбора параметров
# Разделение данных на обучающий и тестовый наборы
train_data = df.iloc[:-12]  # Используем все данные, кроме последних 12 точек для обучения
test_data = df.iloc[-12:]   # Последние 12 точек для тестирования

# Выбор оптимальных параметров SARIMA с помощью автоматического выбора
model = auto_arima(train_data['Значение'], seasonal=True, m=12, trace=True,
                   information_criterion='aic')  # Или information_criterion='bic'

# Подбор параметров SARIMA
result = SARIMAX(train_data['Значение'], order=model.order, seasonal_order=model.seasonal_order).fit()

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

""" 
На основании проведенного анализа временного ряда можно сделать следующие выводы:

По результатам критерия Дики-Фуллера (ADF) полученное значение статистики ADF значительно меньше критических значений,
а p-value равно нулю, что позволяет отвергнуть нулевую гипотезу о нестационарности временного ряда. 
Таким образом, временной ряд стационарен.

На основании автоматического подбора параметров SARIMA была выбрана модель SARIMA(5, 1, 0)(1, 0, 0, 12), 
которая имеет наименьшее значение информационного критерия AIC.


Общий вывод о временном ряде:

Сезонность: Временной ряд обладает сезонными колебаниями с периодом в 12 месяцев. 
Это означает, что данные имеют регулярные циклические колебания, повторяющиеся каждый год.

Уменьшение среднего значения с годами: Обнаружено уменьшение среднего значения временного ряда с течением времени. 
Это можно сделать выводя из убывающих коэффициентов авторегрессии (AR), что указывает 
на уменьшение влияния предыдущих значений на текущее.

Уменьшение частоты колебаний: Также наблюдается уменьшение частоты колебаний ряда. 
Это предположение делается на основе убывающих абсолютных значений коэффициентов авторегрессии (AR), 
что указывает на уменьшение влияния предыдущих значений на текущее и, следовательно, на более плавные изменения в ряду.

Итак, временной ряд характеризуется сезонными колебаниями с периодом в 12 месяцев, 
а также уменьшением среднего значения и уменьшением частоты колебаний с течением времени
"""