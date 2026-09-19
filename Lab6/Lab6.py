import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss,roc_auc_score,roc_curve

# Задание 1
print('------ 1 ------')

wine = pd.read_csv('Wine_Quality_Data.csv')
wine.info()
print(wine.head())
filtered_wine = wine[
    (wine['total_sulfur_dioxide'] <= 300) & 
    (wine['volatile_acidity'] <= 1.4)
]
print(len(filtered_wine) )

X = filtered_wine[['total_sulfur_dioxide', 'volatile_acidity']]
y_true = (filtered_wine['color'] == 'red')


model = LogisticRegression(penalty=None)
model.fit(X, y_true)

probs = model.predict_proba(X)[:, 1]

# Задание 2
print('------ 2 ------')

# Функция матрицы ошибок
def conf_matrix(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    
    tn = sum((y_true == 0) & (y_pred == 0))  # True Negative
    fp = sum((y_true == 0) & (y_pred == 1))  # False Positive
    fn = sum((y_true == 1) & (y_pred == 0))  # False Negative
    tp = sum((y_true == 1) & (y_pred == 1))  # True Positive
    
    return np.array([[tn, fp], [fn, tp]])

# Проверяем три порога
print("\nМатрицы ошибок:")
for threshold in [0.25, 0.5, 0.75]:
    y_pred = (probs >= threshold)
    
    print(f"\nПорог = {threshold}")
    print("Своя функция:")
    print(conf_matrix(y_true, y_pred))
    print("Sklearn:")
    print(confusion_matrix(y_true, y_pred))

# Задание 3
print('\n------ 3 ------')

# Пороги от 0.05 до 0.95 с шагом 0.05
thresholds = np.arange(0.05, 0.96, 0.05)

# Списки для хранения метрик
accuracy_list = []
sensitivity_list = []
specificity_list = []
precision_list = []

# Функция для вычисления метрик по матрице ошибок
def get_metrics(tn, fp, fn, tp):
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0  
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0  
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0  
    return acc, sens, spec, prec

# Расчет метрик для каждого порога
for threshold in thresholds:
    y_pred = (probs >= threshold)
    tn, fp, fn, tp = conf_matrix(y_true, y_pred).ravel()
    
    acc, sens, spec, prec = get_metrics(tn, fp, fn, tp)

    accuracy_list.append(acc)
    sensitivity_list.append(sens)
    specificity_list.append(spec)
    precision_list.append(prec)

plt.plot(thresholds, accuracy_list)
plt.title('Accuracy')
plt.show()
plt.plot(thresholds, sensitivity_list)
plt.title('sensitivity')
plt.show()
plt.plot(thresholds, specificity_list)
plt.title('specificity')
plt.show()
plt.plot(thresholds, precision_list)
plt.title('precision')
plt.show()

# Задание 4
print('\n------ 4 ------')

# Вероятность лежит от 0 до 1
thresholds_roc = np.arange(0, 1.01, 0.01)
fpr_list = []  
tpr_list = []  

for threshold in thresholds_roc:
    y_pred = (probs >= threshold)
    tn, fp, fn, tp = conf_matrix(y_true, y_pred).ravel()
    
    # Истинно отрицательная доля
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    # Истинно положительная доля
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    fpr_list.append(fpr)
    tpr_list.append(tpr)
    

fpr_sklearn, tpr_sklearn, thresholds_sklearn = roc_curve(y_true, probs)
auc_score = roc_auc_score(y_true, probs) 

plt.plot(fpr_list, tpr_list,'r-')
plt.plot(fpr_sklearn, tpr_sklearn,'b-')
plt.show()

# Задание 5
print('\n------ 5 ------')
logloss = log_loss(y_true, probs) # Логистическая потеря
print(logloss)
print(auc_score)

# Задание 6
print('\n------ 6 ------')
X = filtered_wine[['total_sulfur_dioxide', 'volatile_acidity', 'free_sulfur_dioxide']]
y = (filtered_wine['color'] == 'red')

model = LogisticRegression(penalty=None)
model.fit(X, y)

probs = model.predict_proba(X)[:, 1]

logloss = log_loss(y, probs)
auc = roc_auc_score(y, probs)
print(logloss,auc)