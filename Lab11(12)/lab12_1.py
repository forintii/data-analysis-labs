import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


print("-------1-------")

wine = pd.read_csv('Wine_Quality_Data.csv')

le = LabelEncoder()
wine['color'] = le.fit_transform(wine['color'])  # 'red' -> 0, 'white' -> 1

print(wine['color'].sum())

print("-------2-------")

y = wine['quality']
x = wine.drop('quality', axis=1)

print(x.columns)

print("-------3-------")

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.5)

print(f"Обучающая выборка (x_train): {len(x_train)} записей")
print(f"Тестовая выборка (x_test): {len(x_test)} записей")
print(f"Обучающая выборка (y_train): {len(y_train)} записей")
print(f"Тестовая выборка (y_test): {len(y_test)} записей")

print("-------4-------")

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Accuracy на обучающей выборке: {train_acc:.4f}")
print(f"Accuracy на тестовой выборке: {test_acc:.4f}")

print("-------5-------")

def gini(y):
    counts = np.unique(y, return_counts=True)[1]
    s = counts.sum()
    return 1 - ((counts / s) ** 2).sum()

def entropy(y):
    counts = np.unique(y, return_counts=True)[1]
    p = counts / counts.sum()
    return -p @ np.log2(p)


gini_train = gini(y_train)
entropy_train = entropy(y_train)

print(f"gini = {gini_train:.4f}")
print(f"entropy = {entropy_train:.4f}")

print("-------6-------")
# Получаем feature и threshold в корне дерева
feature_index = model.tree_.feature[0]
threshold = model.tree_.threshold[0]
feature_name = x.columns[feature_index]

print(f"{feature_name} <= {threshold:.3f}")

print("-------7-------")

mask_left = x_train[feature_name] <= threshold   # левый узел (True)
mask_right = x_train[feature_name] > threshold  # правый узел (False)

# Классы в левой и правой частях
left_classes = y_train[mask_left]
right_classes = y_train[mask_right]

# Подсчёт количества записей каждого класса
left_counts = left_classes.value_counts()
right_counts = right_classes.value_counts()

print("Левый узел")
for quality, count in left_counts.items():
    print(f"{quality} {count}")

print("Правый узел")
for quality, count in right_counts.items():
    print(f"{quality} {count}")


print("-------8-------")
# Gini для левой части
gini_left = gini(left_classes)
count_left = len(left_classes)

# Gini для правой части
gini_right = gini(right_classes)
count_right = len(right_classes)

# Суммарная Gini impurity 
total_count = len(y_train)
total_gini = (count_left / total_count) * gini_left + (count_right / total_count) * gini_right

print(f"count1 = {count_left} gini1 = {gini_left}")
print(f"count2 = {count_right} gini2 = {gini_right}")
print(f"total count = {total_count} total gini = {total_gini}")
print("-------9-------")
experiments = [
    ('alcohol',   10.0),
    ('alcohol',   11.5),
    ('pH',         3.3),
    ('density',    0.994),
    ('sulphates',  0.6),
]

total_count = len(y_train)

for feat, val in experiments:
    ml = x_train[feat] <= val
    mr = ~ml
    c1, c2 = ml.sum(), mr.sum()
    
    if c1 == 0 or c2 == 0:
        print(f"{feat} <= {val}: деление вырождено")
        continue
    
    
    g1 = gini(y_train[ml])
    g2 = gini(y_train[mr])
    
    tg = (c1 * g1 + c2 * g2) / total_count
    print(f"{feat} <= {val:6}  |  n_left={c1:4d}  n_right={c2:4d}  total_gini={tg:.6f}")

print(f"\nОптимальное ({feature_name} <= {threshold:.3f}) -> total_gini = {total_gini:.6f}")