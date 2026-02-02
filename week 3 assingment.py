import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
iris = datasets.load_iris()

import pandas as pd
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)
weight = PlantGrowth[['weight']]

#question 1 a

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
sw= iris_df['sepal width (cm)']
plt.hist(sw)
plt.show()
#question 1b
print("the mean should be higher as there are more values in the lower quadrent the the upper")

#question 1c

mean=np.mean(sw)
median=np.median(sw)

print("the mean is :", mean)
print("the median is : ", median)

#question 1d

topvalue = sw.quantile(0.73)

print("the value of the top 27% is: ", topvalue)

#question 1e

sl= iris_df['sepal length (cm)']
pl= iris_df['petal length (cm)']
pw= iris_df['petal width (cm)']

plt.scatter(sw, sl)
plt.xlabel('sepal width')
plt.ylabel('sepal length')
plt.title('sepal width vs sepal length')
plt.show()

plt.scatter(sw, pw)
plt.xlabel('sepal width')
plt.ylabel('petal width')
plt.title('sepal width vs petal width')
plt.show()

plt.scatter(sw, pl)
plt.xlabel('sepal width')
plt.ylabel('petal length')
plt.title('sepal width vs petal length')
plt.show()

plt.scatter(sl, pw)
plt.xlabel('sepal length')
plt.ylabel('petal width')
plt.title('sepal length vs petal width')
plt.show()

plt.scatter(sl, pl)
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.title('sepal length vs petal length')
plt.show()

plt.scatter(pw, pl)
plt.xlabel('petal width')
plt.ylabel('petal length')
plt.title('petal width vs petal length')
plt.show()

#question 1f
print("petal width vs petal length seems to have the highest correlation as most of the points are clustered along the mid line.")
print("sepal width vs sepal length seems to have the least correlation as the points are the most spread out.")

#question 2a

start = 3.3
step = 0.3
max_val = float(np.max(weight))
bins = np.arange(start, max_val + step, step)

plt.hist(weight, bins=bins)
plt.title('Histogram of Plant Growth Weights', fontsize=14)
plt.xlabel('Weight', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(bins)
plt.grid(axis='y', alpha=0.3)
plt.show()

#question 2b

sns.boxplot(data=PlantGrowth, x='group', y='weight')
plt.title('Weight distribution')
plt.xlabel('group')
plt.ylabel('weight')
plt.show()

#quesiton 2c
print("it looks like the min is a little less the 5, there are only two outliers above that of the ten,  so 80%.")

#question 2d

min_trt2 = PlantGrowth[PlantGrowth['group'] == 'trt2']['weight'].min()
trt1_below = PlantGrowth[(PlantGrowth['group'] == 'trt1') & (PlantGrowth['weight'] < min_trt2)]
percentage = (len(trt1_below) / len(PlantGrowth[PlantGrowth['group'] == 'trt1'])) * 100
print("the percentage below the min of trt2 is: ", percentage)


#question 2e

highGrowth= PlantGrowth[PlantGrowth['weight']>5.5]
sns.countplot (data= highGrowth, x='group', palette= 'magma')
plt.title('Plants with growth over 5.5 by group')
plt.show()