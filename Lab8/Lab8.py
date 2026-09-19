from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 

print('------ 1 ------')


df = pd.read_csv('Wine_Quality_Data.csv')

red_wines = df[df['color'] == 'red']

print(len(red_wines))

print('------ 2 ------')
# Разбиение
train = red_wines.sample(frac=0.5, random_state=42)
test = red_wines.drop(train.index)

train_q5 = (train['quality'] > 5).sum()
test_q5 = (test['quality'] > 5).sum()

print(f"Обучающая выборка: {train_q5}")
print(f"Тестовая выборка: {test_q5}")

print('------ 3 ------')

train_x = train[['volatile_acidity', 'alcohol']].values
test_x = test[['volatile_acidity', 'alcohol']].values


train_y = (train['quality'] > 5).values
test_y = (test['quality'] > 5).values


print(train_x[:3])
print(test_x[:3])
print(train_y[:3])
print(test_y[:3])


print('------ 4 ------')

accuracies = []
for k_neighbors in [1, 5, 10, 20, 30, 50]:
    
    model = make_pipeline(StandardScaler(), 
                          KNeighborsClassifier(n_neighbors=k_neighbors))
    
    model.fit(train_x, train_y)
    
    y_pred = model.predict(test_x)
    
    acc = accuracy_score(test_y, y_pred)
    accuracies.append(acc)
    print(f"k = {k_neighbors} accuracy = {acc}")


print('------ 5 ------')

k_values = [1, 5, 10, 20, 30, 50]
fig, sub = plt.subplots(3, 2, figsize=(12, 15))

for i, k_nei in enumerate(k_values):
    
    model = make_pipeline(StandardScaler(),
                          KNeighborsClassifier(n_neighbors=k_nei))
    
    
    model.fit(train_x, train_y)
    
    DecisionBoundaryDisplay.from_estimator(
        model, test_x,
        response_method="predict",
        grid_resolution=200,
        cmap=ListedColormap(['cyan', 'cornflowerblue']),
        ax=sub.flatten()[i]
    )
    
    sub.flatten()[i].scatter(test_x[:, 0], test_x[:, 1], 
                              c=test_y, s=2,
                              cmap=ListedColormap(['red', 'green']))
    
    
    sub.flatten()[i].set_title(f'n_neighbors = {k_nei}, acc = {accuracies[i]:.4f}')
plt.show()


print('------ 6 ------')


train_x = train[['volatile_acidity','alcohol', 'sulphates', 'citric_acid']].values  
test_x = test[['volatile_acidity','alcohol', 'sulphates', 'citric_acid']].values 


train_y = (train['quality'] > 5).values
test_y = (test['quality'] > 5).values

accuracies = []
for k_neighbors in [1, 5, 10, 20, 30, 50]:

    model = make_pipeline(StandardScaler(), 
                          KNeighborsClassifier(n_neighbors=k_neighbors))

    model.fit(train_x, train_y)
    

    y_pred = model.predict(test_x)
    
    acc = accuracy_score(test_y, y_pred)
    accuracies.append(acc)
    print(f"k = {k_neighbors} accuracy = {acc}")