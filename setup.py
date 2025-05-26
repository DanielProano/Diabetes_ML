from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
import matplotlib.pyplot as plt

#Load the data
diabetes = load_diabetes()

#Make the DataFrame, a 2D array basically
diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

#Make the target column for our values
diabetes_df['target'] = diabetes.target

#Now we need to split the data temporarily
input_d = diabetes_df.drop('target', axis=1)

output = diabetes_df['target']

#Now organize the data
X_train, X_test, y_train, y_test = train_test_split(input_d, output, train_size=0.8, random_state=10)

#Using the Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)

X_predict = model.predict(X_test)

l_score = model.score(X_test, y_test)
print(f"Linear Regression result {l_score}")

#Using Ridge Model
r_model = Ridge(alpha=1.0)

r_model.fit(X_train, y_train)

r_predict = r_model.predict(X_test)

r_score = r_model.score(X_test, y_test)
print(f"Ridge Model score result {r_score}")

plt.scatter(y_test, X_predict)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()





