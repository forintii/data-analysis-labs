
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

print('------ 1 ------')
# Чтение данных
df = pd.read_csv('Wine_Quality_Data.csv')

# Применение фильтров
filtered_df = df[
    (df['total_sulfur_dioxide'] <= 250) &
    (df['volatile_acidity'] <= 1.4) &
    (df['quality'] > 6)
]

# Вывод количества записей
print(len(filtered_df))

print('------ 2 ------')

# Признаки и целевая переменная (на отфильтрованных данных)
X = filtered_df[['total_sulfur_dioxide', 'volatile_acidity']]
y = filtered_df['color'] == 'red'

# Модель SVM с линейным ядром
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

# Вывод коэффициентов разделяющей прямой
print(model.coef_, model.intercept_)


print('------ 3 ------')

# Отображение границ классификации
DecisionBoundaryDisplay.from_estimator(
    model, X,
    response_method="predict",
    grid_resolution=200,
    cmap=ListedColormap(['cyan', 'cornflowerblue'])
)

# Отображение точек выборки
plt.scatter(X.iloc[:,0], X.iloc[:,1],
           c=[['y','r'][i] for i in y], s=2)


plt.show()

print('------ 4 ------')
X_sup      = X[model.support_]
y_sup      = y[model.support_]
y_sup_pred = model.predict(X_sup)

correct   = y_sup == y_sup_pred
incorrect = ~correct

fig, ax = plt.subplots(figsize=(10, 7))
# Отображение границ классификации
DecisionBoundaryDisplay.from_estimator(
    model, X,
    response_method="predict",
    grid_resolution=200,
    cmap=ListedColormap(['cyan', 'cornflowerblue']),
    ax=ax
)
ax.scatter(X.iloc[:,0], X.iloc[:,1],
           c=[['y','r'][i] for i in y], s=2)

# Верно классифицированные → треугольники
ax.scatter(X_sup[correct, 0], X_sup[correct, 1],
           c=[['y','r'][i] for i in y],
           marker='^', s=100, edgecolors='black', linewidths=1,
           label='Опорные: верно (треугольник)')

# Неверно классифицированные → крестики
ax.scatter(X_sup[incorrect, 0], X_sup[incorrect, 1],
           c=[['y','r'][i] for i in y],
           marker='x', s=100,
           label='Опорные: неверно (крестик)')


plt.savefig('task4.png')
print(f"Количество опорных векторов: {len(X_sup)}")
print(f"  верно классифицированных:   {correct.sum()}")
print(f"  неверно классифицированных: {incorrect.sum()}")