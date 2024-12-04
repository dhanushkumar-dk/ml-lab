from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

training_data = [
    ['Chinese Beijing Chinese', 'china'],
    ['Chinese Chinese Shanghai', 'china'],
    ['Chinese Macao', 'china'],
    ['Tokyo Japan Chinese', 'japan']
]

X = [i[0] for i in training_data]
y = [i[1] for i in training_data]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)

clf = MultinomialNB()

clf.fit(X, y)

test_data = ['Chinese Chinese Chinese Tokyo Japan', 'Beijing China']
test_labels = ['japan', 'china']

test_data = vectorizer.transform(test_data)

predictions = clf.predict(test_data)

accuracy = accuracy_score(test_labels, predictions)
print("Accuracy: ", accuracy)
