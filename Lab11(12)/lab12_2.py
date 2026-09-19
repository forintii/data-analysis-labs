import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

print("-------4-------")
# Создание и обучение модели (без ограничений)
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)


# Получаем соответствие индексов и классов
classes = np.unique(y_train)  # [3, 4, 5, 6, 7, 8]

def Mypredict(tree, X):
    X_list = X.values.tolist()
    predictions = []
    
    for row in X_list:
        node = 0
        while tree.feature[node] != -2:
            feat_idx = tree.feature[node]
            thr = tree.threshold[node]
            
            if row[feat_idx] <= thr:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]
        
        max_index = 0
        max_count = 0
        for i, count in enumerate(tree.value[node][0]):
            if count > max_count:
                max_count = count
                max_index = i
        
        predictions.append(classes[max_index])
    
    return np.array(predictions)

# Делаем предсказания
y_train_pred = Mypredict(model.tree_, x_train)
y_test_pred = Mypredict(model.tree_, x_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Accuracy на обучающей выборке: {train_acc:.4f}")
print(f"Accuracy на тестовой выборке: {test_acc:.4f}")
