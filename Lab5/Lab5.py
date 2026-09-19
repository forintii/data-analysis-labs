import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import sklearn.datasets as ds
from sklearn.linear_model import LogisticRegression

# Задание 1
wine = pd.read_csv('Wine_Quality_Data.csv')

print("Задание 1")
print(wine.info())
print(wine.head())

wine = wine[
    (wine['total_sulfur_dioxide'] <= 300) &
    (wine['volatile_acidity'] <= 1.4)
]

# Подсчёт количества записей
count = wine.shape[0]
print(count)

# Задание 2 
print('Задание 2')
red = wine[wine['color'] == 'red']
white = wine[wine['color'] == 'white']

plt.hist([red['total_sulfur_dioxide'],white['total_sulfur_dioxide']],color=['r','y'],bins=30)
plt.show()
plt.hist([red['volatile_acidity'],white['volatile_acidity']],color=['r','y'],bins=30)
plt.show()

# Задание 3
print('Задание 3')

X1 = wine[['total_sulfur_dioxide']]
y = wine['color'] == 'red'

model1 = LogisticRegression(penalty=None)
model1.fit(X1, y)
print("Коэффициент (наклон):", model1.coef_[0])
print("Свободный член:", model1.intercept_[0])

X2 = wine[['volatile_acidity']]

model2 = LogisticRegression()
model2.fit(X2, y)
print("Коэффициент (наклон):", model2.coef_[0][0])
print("Свободный член:", model2.intercept_[0])

# Задание 4
print('Задание 4')

ptx = np.linspace(0, 300, 100).reshape(-1, 1)
probs1 = model1.predict_proba(ptx)[:, 1]
print(probs1)