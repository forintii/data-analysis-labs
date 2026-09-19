import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import sklearn.datasets as ds

#  1. Загрузите таблицу с данными из файла avocado.csv Выполните фильтрацию данных по столбцу 
#  'type', оставив строки со значением 'conventional' и столбцы 'region', 'AveragePrice', 
#  'Total Volume'. Переименуйте столбцы в 'region', 'price', 'volume'. Выведите статистику, 
#   используя метод info().
print('------ 1 ------')
avo=pd.read_csv('avocado.csv')
avo_filter = avo.loc[avo['type'] == 'conventional', ['region', 'AveragePrice', 'Total Volume']].copy()
avo_filter.columns = ['region', 'price', 'volume']
print(avo_filter.info())
print(avo_filter.head())


#  2. Постройте модель парной регрессии для для предсказания объема продаж по цене, используя
#  класс LinearRegression. Выведите уравнение регрессии, используя коэффициенты, найденные в модели.
#  Оцените качество модели по коэффициенту детерминации. Найдите среднеквадратическую ошибку, используя 
#  функцию mean_squared_error. Постройте график уравнения регрессии, используя
#  найденное уравнение.
#  Нанесите на график точки, соответствующие измерениям цены и объема продаж.
print('------ 2 ------')
X = pd.DataFrame(avo_filter['price'])
y = avo_filter['volume']

model = LinearRegression()
model.fit(X, y)

a1 = model.coef_
a0 = model.intercept_
print(f"Уравнение регрессии y = {a0} + {a1} * X")

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"R2 = {r2}")
print(f"MSE = {mse}")

linex = np.linspace(X.min(),X.max(),2)
liney = a1*linex+a0
plt.scatter(X, y, s=5)
plt.plot(linex,liney,c='r')
plt.show()
#  3. Напишите функции для нахождения коэффициентов уравнения регрессии (аналог fit),
#  для нахождения предсказанных значений (аналог predict), для нахождения
#  коэффициента детермигации (аналог r2_score) и для нахождения среднеквадратической ошибки
#  Повторите решение задания предыдущего пункта, используя написанные самостоятельно функции.
print('------ 3 ------')

def my_fit(X, y):
    X = pd.DataFrame({'ones': np.ones(len(X)), 'f1': X.iloc[:, 0]})
    a = np.linalg.inv(X.T@X)@X.T@y
    return a, X
b, X1 = my_fit(X, y)
print(b)
print(f"Уравнение регрессии y = {b[0]} + {b[1]} * X")

def my_predict(X, a):
    pred_y = X@np.array(a)
    return pred_y
pred_y = my_predict(X1,b)

def my_r2(y, pred_y):
    r2 = 1-((y-pred_y)**2).sum()/((y-y.mean())**2).sum()
    return r2
r2_my = my_r2(y, pred_y)
print(f"R2 = {r2_my}")

def my_mean_squared_error(y, pred_y):
    loss = ((y-pred_y)**2).sum()/ len(y)
    return loss
loss = my_mean_squared_error(y,pred_y)
print(f"MSE = {loss}")

linex = np.linspace(X.min(),X.max(),2)
liney = a1*linex+a0
plt.scatter(X, y, s=5)
plt.plot(linex,liney,c='r')
plt.show()

#  4. Найдите средний объем продаж для каждого региона. Добавьте в таблицу столбец 'meanvol' 
#  со средним объемом продаж. Выведите статистику, используя метод info().
#  Постройте модель регреccии с факторами 'price' и 'meanvol'. 
#  Найдите коэффициент детерминации и среднеквадратическую ошибку.
print('------ 4 ------')

avo_filter['meanvol'] = avo_filter.groupby('region')['volume'].transform('mean')
print(avo_filter.info())
print(avo_filter[['region', 'price', 'volume', 'meanvol']].head())

model2 = LinearRegression()
X2 = avo_filter[['price', 'meanvol']]
model2.fit(X2, y)


y_pred = model2.predict(X2)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"R2 = {r2}")
print(f"MSE = {mse}")



#  5. Уберите из модели фактор 'price'. Проверьте, как делает предсказание эта модель. Для этого
#  добавьте в данные столбец с предсказанными значениями и выведите каждую пятисотую строку матрицы.
#  Проверьте коэффициент детерминации и среднеквадратическую ошибку. Сделайте вывод.
print('------ 5 ------')

model3 = LinearRegression()
X3 = avo_filter[['meanvol']]
model3.fit(X3, y)

y_pred = model3.predict(X3)
avo_filter['predicted_volume'] = y_pred

print(avo_filter.iloc[::500][['region', 'price', 'meanvol', 'volume', 'predicted_volume']].to_string())

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
print(f"R2 = {r2}")
print(f"MSE = {mse}")

# Основным фактором, средним объемом продаж (meanvol).





