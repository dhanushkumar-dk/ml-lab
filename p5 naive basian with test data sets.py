from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Sample training data
# Assume the data is in the format [features, label]
training_data = [
    ['Chinese Beijing Chinese', 'china'],
    ['Chinese Chinese Shanghai', 'china'],
    ['Chinese Macao', 'china'],
    ['Tokyo Japan Chinese', 'japan']
]

# Prepare the data for training
X = [i[0] for i in training_data]
y = [i[1] for i in training_data]

# Initialize the vectorizer
vectorizer = CountVectorizer()

# Transform the training data using the vectorizer
X = vectorizer.fit_transform(X)

# Initialize the classifier
clf = MultinomialNB()

# Train the classifier
clf.fit(X, y)

# Sample test data
test_data = ['Chinese Chinese Chinese Tokyo Japan', 'Beijing China']
test_labels = ['japan', 'china']

# Transform the test data using the vectorizer
test_data = vectorizer.transform(test_data)

# Make predictions
predictions = clf.predict(test_data)

# Compute the accuracy
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy: ", accuracy)
