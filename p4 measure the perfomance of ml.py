import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

# Load dataset
dataframe = pd.read_csv("diabetes.csv")
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Evaluate using a train and test set
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Evaluate the model
result = model.score(X_test, Y_test)
print("Evaluating using Train and Test sets")
print(f"Accuracy: {result * 100.0:.3f}%")

# Evaluate using Leave-One-Out Cross Validation (LOOCV)
loocv = model_selection.LeaveOneOut()
model = LogisticRegression(max_iter=1000)
results = model_selection.cross_val_score(model, X, Y, cv=loocv)

print("\nEvaluating using Leave-One-Out Cross Validation")
print(f"Accuracy: {results.mean() * 100.0:.3f}% ({results.std() * 100.0:.3f}%)")
