import pandas as pd
import matplotlib.pyplot as plt

avo=pd.read_csv('Lab3/avocado.csv')
print(avo.head())
print(avo.info())

# 1. Пропуски в данных. На первый взгляд пропусков нет. Однако, надо
# проверить, верно ли, что для каждого региона, каждого вида авокадо и даты
# в таблице есть запись.
# Найдите количество различных дат в данных. Выведите полученное число
# Сгруппируйте данные по регионам и видам авокадо и постройте таблицу в которой
# для каждого региона и вида авокадо записано количество записей каждого вида.
# Найдите регионы в которых количество записей меньше чем количество дат.
# Выведите список найденных регионов.
print('------ 1 ------')
print(avo['Date'].unique().size)
#записи которые относяться к одному региону одному виду
tbl = avo.groupby(['region','type'],as_index=False).agg({'Date':'count'})
print(tbl[tbl['Date'] != 169])



#2. Для найденного региона и вида авокадо найдите все отсутствующие даты
# Для этого надо отобрать все существующие даты для найденного региона
# и вида авокадо и создать из них множество (set)
# Далее создать Series из всех различных дат в данных. Можно использовать 
# метод unique() или индексы из таблицы, возвращенной методом value_counts().
# Далее можно использовать метод isin из класса Series. Оператор ~ позволяет сделать
# поэлементное отрицание для Series из логических значений. Возможны и другие решения.
# Формат вывода может отличаться.
print('------ 2 ------')

reg = avo[(avo['region'] == 'WestTexNewMexico') & (avo['type'] == 'organic')]
reg_date = set(reg['Date'].unique())
unique_date = pd.Series(avo['Date'].unique())
ans = unique_date[~unique_date.isin(reg_date)]
print(ans)


#3. Для каждой даты и вида авокадо найдите минимальную, максимальную
# и среднюю цены  по регионам (в таблице 5 колонок: Date, type, MinPrice, MaxPrace, MeanPrice)
# Переименовать столбцы можно при помощи свойства columns.
# Для проверки выведите первые пять записей при помощи метода head()
print('------ 3 ------')
prices = avo.groupby(['Date','type'], as_index=False).agg({'AveragePrice' : ['min','max','mean']})
prices.columns = ['Date', 'type', 'MinPrice', 'MaxPrice', 'MeanPrice']
print(prices.head())


#4. На основе полученной таблицы из предыдущего задания сделайте таблицу
# в которой индексом будет дата и будет 6 колонок в которых для каждой даты
# будет записана максимальная, минимальная и средняя цена для conventional
# авокадо и для organic авокадо.
# Сначала можно разбить таблицу на две по полю type. Далее можно
# использовать метод set_index для установки столбца 'Date' в качестве индекса.
# Объединить таблицы можно при помощи метода join. Используйте параметры lsuffix и rsuffix.
# Для проверки выведите первые пять записей таблицы.
print('------ 4 ------')
prices.set_index('Date', inplace=True)
conv = prices[prices['type'] == 'conventional'].copy()
org = prices[prices['type'] == 'organic'].copy()
conv.drop(columns='type',inplace=True)
org.drop(columns='type', inplace=True)
prices = conv.join(org,lsuffix='Conv',rsuffix='Org')
print(prices.head())


#5.  Постройте графики изменения цен на авокадо (В одном окне 6 графиков)
print('------ 5 ------')
plt.plot(prices)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))  # максимум 8 подписей
plt.legend(prices.columns, loc='upper right')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# 6. Создайте таблицу, в которой для каждой даты будет хранится суммарный объем продаж по всем регионам
# для каждого типа авокадо. В таблице должно быть два столбца: Total Volume Org и Total Volume Conv.
# Дата должна быть индексом. Для проверки выведите графики изменения суммарного объема продаж по датам.
print('------ 6 ------')
vol = avo.groupby(['Date','type'],as_index=False).agg({'Total Volume' : 'sum'})
vol.set_index(['Date'], inplace=True)
conv = vol[vol['type'] == 'conventional'].copy()
org = vol[vol['type'] == 'organic'].copy()
conv.drop(columns='type',inplace=True)
org.drop(columns='type', inplace=True)
vol = conv.join(org,lsuffix='Conv',rsuffix='Org')
print(vol)
plt.plot(vol)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))  # максимум 8 подписей
plt.legend(vol.columns, loc='upper right')
plt.show()

# 7. Создайте таблицу, в которой для каждого региона и типа авокадо будет хранится максимальная
# стоимость только по тем дням, когда объем продаж был выше среднего. Для решения задачи используйте
# метод apply. Для проверки выведите первые восемь записей таблицы.
print('------ 7 ------') 
def max_price_above_avg(group): 
    avg_volume = group['Total Volume'].mean() 
    above_avg = group[group['Total Volume'] > avg_volume] 
    return above_avg['AveragePrice'].max() 
 
result = avo.groupby(['region', 'type']).apply(max_price_above_avg)
result.columns = ['region', 'type', 'max_price'] 
print(result.head(8)) 
# 8. Отберите из набора данных строки с type==conventional. Сгруппируйте данные по дате. Для 
# каждой даты найтите среднюю, медианную и средневзвешенную цену на авокадо.
# Постройте графики изменения цен.
print('------ 8 ------') 
conv = avo[avo['type'] == 'conventional'].copy() 
 
def weighted_avg(group): 
    weights = group['Total Volume'] 
    values = group['AveragePrice'] 
    return (values * weights).sum() / weights.sum() 
 
daily_stats = conv.groupby('Date').agg( 
    mean_price=('AveragePrice', 'mean'), 
    median_price=('AveragePrice', 'median')).round(3) 
 
weighted_prices = conv.groupby('Date').apply(weighted_avg).round(3) 
daily_stats['weighted_mean'] = weighted_prices 
 
print(daily_stats.head())
plt.plot(daily_stats)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))  # максимум 8 подписей
plt.legend(vol.columns, loc='upper right')
plt.show()
