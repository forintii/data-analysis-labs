import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("-------1-------")
# 1. Чтение данных
wine = pd.read_csv('Wine_Quality_Data.csv')

# 2. Преобразование 'color' в числовой формат
le = LabelEncoder()
wine['color'] = le.fit_transform(wine['color'])  # 'red' -> 0, 'white' -> 1

# 3. Вывод суммы
print(wine['color'].sum())

print("-------2-------")

# Выделение целевой переменной и признаков
y = wine['quality']
x = wine.drop('quality', axis=1)

# Вывод списка столбцов x
print(x.columns)

print("-------3-------")
# Разбиение данных (как указано в задании)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.5)

# Вывод количества записей
print(f"Обучающая выборка: {len(x_train)} записей")
print(f"Тестовая выборка: {len(x_test)} записей")

print("-------5-------")

# список значений n_estimators
estimators_list = [1, 5, 20, 50, 100, 200]

for n in estimators_list:
    model = RandomForestClassifier(n_estimators=n,random_state=42,bootstrap=True)
    # обучение модели
    model.fit(x_train, y_train)

    # предсказания
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # вычисление accuracy
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # разность между истинными и предсказанными значениями
    diff = y_test - y_pred_test

    # количество ошибок больше 1
    deviation = (abs(diff) > 1).sum()

    # вывод результатов
    print(f'n_estimators = {n}')
    print(f'train accuracy = {train_acc}')
    print(f'test accuracy = {test_acc}')
    print(f'deviation>1 : {deviation}')
    print()

    # столбчатая диаграмма
    diff.value_counts().sort_index().plot(kind='bar')
    plt.title(f'Difference distribution (n_estimators={n})')
    plt.xlabel('y_test - y_test_pred')
    plt.ylabel('Count')
    plt.show()

print("-------8-------")
for n in estimators_list:
    model = RandomForestClassifier(n_estimators=n,random_state=42,bootstrap=False)
    # обучение модели
    model.fit(x_train, y_train)

    # предсказания
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # вычисление accuracy
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # разность между истинными и предсказанными значениями
    diff = y_test - y_pred_test

    # количество ошибок больше 1
    deviation = (abs(diff) > 1).sum()

    # вывод результатов
    print(f'n_estimators = {n}')
    print(f'train accuracy = {train_acc}')
    print(f'test accuracy = {test_acc}')
    print(f'deviation>1 : {deviation}')
    print()

print("-------9-------")

max_features_list = [1, 2, 4, 6, 'sqrt', None]

for mf in max_features_list:

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_features=mf)

    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f'max_features = {mf}')
    print(f'train accuracy = {train_acc}')
    print(f'test accuracy = {test_acc}')
    print()