## Lab11.1 — Дерево решений и критерии информативности (Gini, Entropy)

**Файл:** `Lab12_1.py`, `Wine_Quality_Data.csv`

**Что делали:**
- Загрузка данных о вине, кодирование категориального признака `color` через `LabelEncoder`
- Разделение на признаки (`x`) и целевую переменную (`y = quality`)
- Разбиение на train/test (50/50) через `train_test_split`
- Построение **дерева решений** (`DecisionTreeClassifier`) без ограничений
- Оценка accuracy на обучающей и тестовой выборках
- **Собственная реализация критериев Gini и Entropy** через `numpy`
- Анализ **корневого узла дерева**: извлечение признака и порога (`model.tree_.feature`, `model.tree_.threshold`)
- Ручное разбиение данных по корневому порогу, подсчёт классов в левом и правом узлах
- **Расчёт суммарной Gini impurity** для разбиения
- Сравнение с альтернативными порогами по разным признакам (`alcohol`, `pH`, `density`, `sulphates`)

**Стек:** Python, pandas, numpy, sklearn

**Как запустить:**
```bash
cd Lab11\(12\)
python Lab12_1.py
```

## Lab11.2 — Собственная реализация predict для дерева решений

**Файл:** `Lab12_2.py`, `Wine_Quality_Data.csv`

**Что делали:**
- Подготовка данных: кодирование `color`, разделение на `x` и `y`, train/test split
- Построение дерева решений (`DecisionTreeClassifier`)
- **Собственная реализация функции `Mypredict`**: обход дерева вручную
  - Проход по узлам через `tree.feature` и `tree.threshold`
  - Определение листового узла (`feature == -2`)
  - Выбор класса с наибольшим количеством объектов в листе
- Сравнение accuracy собственной реализации с библиотечной

**Стек:** Python, pandas, numpy, sklearn

**Как запустить:**
```bash
cd Lab11\(12\)
python Lab12_2.py
```

## Lab11.3 — Случайный лес (Random Forest) и подбор гиперпараметров

**Файл:** `Lab13.py`, `Wine_Quality_Data.csv`

**Что делали:**
- Подготовка данных: кодирование `color`, разделение на `x` и `y`, train/test split
- Построение **случайного леса** (`RandomForestClassifier`) с разными `n_estimators` (1, 5, 20, 50, 100, 200)
- Оценка accuracy на train и test
- **Анализ распределения ошибок**: `diff = y_test - y_pred`, подсчёт ошибок > 1
- Визуализация распределения ошибок (столбчатые диаграммы) для каждого `n_estimators`
- **Сравнение с `bootstrap=False`** — проверка, как влияет бутстрэп на качество
- **Подбор `max_features`** (1, 2, 4, 6, 'sqrt', None) и оценка accuracy

**Стек:** Python, pandas, numpy, sklearn, matplotlib

**Как запустить:**
```bash
cd Lab11\(12\)
python Lab13.py
```
