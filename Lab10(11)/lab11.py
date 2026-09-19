import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Задание 1
# Загружаем набор данных Iris
x, y = load_iris(return_X_y=True, as_frame=True)

# Выводим информацию для проверки
print("Размерность признаков:", x.shape)
print("Размерность целевой переменной:", y.shape)
print("Уникальные классы:", y.unique())
print()

# Выводим по одному примеру для каждого класса
print("Класс 0 (Setosa):")
print(x[y == 0].head(1))

print("Класс 1 (Versicolor):")
print(x[y == 1].head(1))

print("Класс 2 (Virginica):")
print(x[y == 2].head(1))


# Задание 2
print("\nЗадание 2")


X2 = x[['sepal length (cm)', 'petal length (cm)']].values

gamma_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

cmap_bg     = ListedColormap(['yellow', 'cyan', 'lightgreen'])
cmap_points = ['red', 'green', 'blue']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, gamma_val in zip(axes.flatten(), gamma_values):
    model = make_pipeline(
        StandardScaler(),
        SVC(gamma=gamma_val, kernel='rbf', C=1)
    )
    model.fit(X2, y)

    y_pred = model.predict(X2)
    acc = accuracy_score(y, y_pred)

    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=cmap_bg, alpha=0.5)

    for cls in [0, 1, 2]:
        mask = y == cls
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=cmap_points[cls], s=20, alpha=0.8,
                   label=f'класс {cls}')

    ax.set_xlabel('sepal length (cm)')
    ax.set_ylabel('petal length (cm)')
    ax.set_title(f'RBF  gamma={gamma_val}  accuracy={acc:.3f}')
    ax.legend(fontsize=7)

plt.suptitle('RBF ядро — влияние параметра gamma', fontsize=14)
plt.tight_layout()
plt.show()


# Задание 3
print("\nЗадание 3")


degrees = [2, 3, 4, 5]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, degree in zip(axes.flatten(), degrees):
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='poly', degree=degree, C=1, coef0=1)
    )
    model.fit(X2, y)

    y_pred = model.predict(X2)
    acc = accuracy_score(y, y_pred)

    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=cmap_bg, alpha=0.5)

    for cls in [0, 1, 2]:
        mask = y == cls
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=cmap_points[cls], s=20, alpha=0.8,
                   label=f'класс {cls}')

    ax.set_xlabel('sepal length (cm)')
    ax.set_ylabel('petal length (cm)')
    ax.set_title(f'poly  degree={degree}  accuracy={acc:.3f}')
    ax.legend(fontsize=7)

plt.suptitle('Полиномиальное ядро — влияние степени', fontsize=14)
plt.tight_layout()
plt.show()


# Задание 4
print("\nЗадание 4")

X4 = x.values
best_model = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', gamma=1, C=100)
)
best_model.fit(X4, y)

y_pred_4 = best_model.predict(X4)
acc_4 = accuracy_score(y, y_pred_4)
print(f"Accuracy на всех 4 признаках: {acc_4:.3f}")


# Задание 5
print("\nЗадание 5")

df_lib = best_model.decision_function(X4)
print("Результат библиотечного decision_function (первые 5 строк):")
print(df_lib[:5])
print(f"Форма результата: {df_lib.shape}")

