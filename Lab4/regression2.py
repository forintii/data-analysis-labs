import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

avo=pd.read_csv('avocado.csv')
#  1. Загрузите таблицу с данными из файла avocado.csv Выполните фильтрацию данных по столбцу 
#  'type', оставив строки со значением 'conventional' и столбцы 'region', 'AveragePrice', 
#  'Total Volume'. Переименуйте столбцы в 'region', 'price', 'volume'. Выведите статистику, 
#   используя метод info().
print('------ 1 ------')
tbl = avo[avo['type']=='conventional'].filter(items=['region','AveragePrice','Total Volume'])
tbl.columns = ['region', 'price', 'volume']
tbl.info()

#   2. Найдите средний объем продаж для каждого региона. Добавьте в таблицу столбец 'meanvol' 
#   со средним объемом продаж. Выведите статистику, используя метод info().
#   Добавьте в таблицу столбец 'kvol' равный 'volume'/'meanvol'. 
#   Что показывает величина в этом столбце?
print('------ 2 ------')
meanvol = tbl.groupby(['region'], as_index=True).agg(meanvol = pd.NamedAgg('volume','mean'))
tbl = tbl.join(meanvol, on='region')
tbl.info()

tbl['kvol'] = tbl['volume']/tbl['meanvol']
print(tbl.head())
# Срдений объем продаж мы определили как условный размер региона, следовательно 'kvol' позволяет
# отобразить долю объема продаж по данной цене в регионе

#  3. Постройте модель парной регрессии для предскзания kvol по цене. 
#  Выведите уравнение регрессии, используя коэффициенты, найденные в модели.
#  Оцените качество модели по коэффициенту детерминации и MSE 
#  Постройте график уравнения регрессии, используя найденное уравнение.
#  Нанесите на график точки, соответствующие измерениям цены и 'kvol'.
print('------ 3 ------')
X = tbl['price']
y = tbl['kvol']

model = LinearRegression()
model.fit(pd.DataFrame(X),y)
print(f"уравнение регрессии y = {model.intercept_} + {model.coef_[0]} * x")

pred_y = model.predict(pd.DataFrame(X))
print('R^2 = ', r2_score(y, pred_y))
print('MSE = ', mean_squared_error(y, pred_y))
# Коэффициент детерминации очень низкий, следовательно влияние цены невелико, модель по фактору цены не точна

_, ax1 = plt.subplots()
linex = np.linspace(X.min(), X.max(), 2)
liney = model.coef_[0]*linex + model.intercept_
plt.plot(linex, liney, c = 'r')
plt.scatter(X, y, s = 2)
plt.show()

#  4. Постройте несколько моделей нелинейной регрессии. Оцените их качество 
#  по коэффициенту детерминации и MSE. 
#  Выведите уравнения регрессии, используя коэффициенты, найденные в модели.
#  Постройте графики уравнения регрессии, используя найденные уравнения.
#  Нанесите на графики точки, соответствующие измерениям цены и 'kvol'.
print('------ 4 ------')


#  5. Для каждого региона найдите средневзвешенную цену и добавьте ее в таблицу с данными. 
#  Добавьте этот фактор в модель линейной регрессии. Проверьте коэффициент детерминации и MSE.
print('------ 5 ------')

#  6. Попробуйте подобрать нелинейную модель (не только полиномиальную) с наилучшим качеством предсказания.
print('------ 6 ------')
