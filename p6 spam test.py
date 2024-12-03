# Importing libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_openml

# Load the spam dataset (from OpenML)
print("Fetching the dataset...")
spam_data = fetch_openml("spambase", version=1, as_frame=True)

# Extract features (X) and labels (y)
X = spam_data.data
y = spam_data.target.astype(int)  # Convert target to integer

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
nb_classifier = MultinomialNB()

# Train the classifier on the training data
print("Training the Naive Bayes classifier...")
nb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the Naive Bayes Classifier: {accuracy:.2f}")

# Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example prediction
sample_test = X_test.iloc[:1]  # Taking the first test sample
sample_pred = nb_classifier.predict(sample_test)
print(f"\nSample Test Prediction: {'Spam' if sample_pred[0] == 1 else 'Not Spam'}")
