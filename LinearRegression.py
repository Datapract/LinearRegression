import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\miscasrikanth\Desktop\Data science\Refactored_Py_DS_ML_Bootcamp-master\11-Linear-Regression\USA_Housing.csv')


print(df.head())
print(df.info())
print(df.describe())
print(df.columns)



# sns.pairplot(df)
# sns.displot(df['Price'])
# sns.heatmap(df.corr(),annot=True)
# plt.show()

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]

Y = df[['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, Y_train)

# print(lm.intercept_)
# print(lm.coef_)

cdf = pd.DataFrame(lm.coef_.transpose(), X.columns, columns=['Coeff'])

print(cdf)

predictions = lm.predict(X_test)

print(predictions)

#plt.scatter(Y_test,predictions)
#sns.distplot((Y_test-predictions))
#plt.show()

from sklearn import metrics

print('absolute_error-->',metrics.mean_absolute_error(Y_test,predictions))
print('squared_error-->',metrics.mean_squared_error(Y_test,predictions))
print('RootMeanSquare_error-->',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))

