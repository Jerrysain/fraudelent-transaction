# fraudelent-transaction
Fraud detection: Build a system that can detect fraudulent behaviour in financial transactions, such as credit card transactions.
# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Exploratory data analysis
print('Number of transactions:', len(data))
print('Number of fraudulent transactions:', len(data[data['Class'] == 1]))
print('Number of non-fraudulent transactions:', len(data[data['Class'] == 0]))
print('Percentage of fraudulent transactions:', len(data[data['Class'] == 1]) / len(data) * 100, '%')

# Visualize the distribution of transaction amount
sns.histplot(data=data, x='Amount', hue='Class', bins=20, kde=True)
plt.show()

# Scale the features
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time'], axis=1)

# Split the dataset into training and testing sets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the class labels for the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)
print('Confusion matrix:', confusion_matrix)


In this code, we first load the dataset using Pandas library. We then perform exploratory data analysis to understand the distribution of fraudulent transactions and visualize the distribution of transaction amounts using the Seaborn library. We then scale the features using the StandardScaler class of scikit-learn.

Next, we split the dataset into training and testing sets using the train_test_split function. We fit a logistic regression model on the training set using scikit-learn's LogisticRegression class. We then predict the class labels for the test set using the predict method of the logistic regression model.

Finally, we evaluate the model's performance using various metrics such as accuracy, precision, recall, F1 score, and confusion matrix. These metrics help us understand how well the model is performing in terms of detecting fraudulent transactions.
