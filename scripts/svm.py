import json
import pandas as pd
import numpy as np
import pickle
from string import punctuation
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

file="HW3/yelp_dataset/yelp_academic_dataset_review.json"
f = open('svm_report.txt', 'w')

# read the first 500,000 yelp reviews
lines=open(file, encoding="utf8").readlines()[:500000]

data = []
for line in lines:
    data.append(json.loads(line))
content, label = [], []
for each in data:
    content.append(each.get('text'))
    label.append(each.get('stars'))
    
df = pd.DataFrame([content, label]).T
df.columns= ['review', 'stars']

def process_text(text):
    # apply a function that strips punctuation and converts to lower case.
    tokens = map(lambda x: x.strip(punctuation).lower(), text.split()) 
    tokens = list(filter(None, tokens))
    return tokens

#Apply the function to preprocess the text. Tokenize, lower, remove punctuation, numbers and stop words
df['clean_text'] = df['review'].apply(process_text)
print(f"Input corpus sample:\n{df.head()}", file=f)

#Turning the labels into numbers
LE = LabelEncoder()
df['stars_num'] = LE.fit_transform(df['stars'])
print(f"\nEncoded Labels: {df['stars_num'].unique()}", file=f)

# Train test split with stratified sampling for evaluation
X = df['clean_text']
y = df['stars_num']
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = .2, 
                                                    shuffle = True, 
                                                    stratify = y, 
                                                    random_state = 42)

# Verify distribution of stars in train and test
stars, counts = np.unique(y_train, return_counts=True)
print(f'\nDistribution of labels in trianing data:\n{stars} {counts/sum(counts)}', file=f)

stars, counts = np.unique(y_test, return_counts=True)
print(f'Distribution of labels in test data:\n{stars} {counts/sum(counts)}', file=f)

model_perf = pd.DataFrame(columns=['ngram', 'model', 'parameters', 'accuracy', 'precision', 'recall', 'F1'])

"""Unigrams"""
#Creating the features (tf-idf weights) for the processed text
texts = X_train.astype('str')

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1),
                                   min_df = 5, 
                                   max_df = .5,
                                   max_features = 100000)
X_train_vec = tfidf_vectorizer.fit_transform(texts) #features
pickle.dump(tfidf_vectorizer, open("unigram_vectorizer.pickle", "wb"))

y_train_vec = y_train.values #target

print(f"\nDimensions of X_train with unigrams = {X_train_vec.shape}", file=f)
print(f"Dimensions of y_train = {y_train_vec.shape}", file=f)

# Fitting Logistic Regression model
X_test_vec = tfidf_vectorizer.transform(X_test.astype('str'))

clf = LinearSVC(penalty='l2', multi_class='ovr', class_weight='balanced', random_state=42)
clf.fit(X_train_vec, y_train_vec)
pickle.dump(clf, open("svm_model_1.sav", 'wb'))

y_pred = clf.predict(X_test_vec)
model_perf = model_perf.append({'ngram':'uni',
                                'model': 'svm', 
                                'parameters': {'multi_class':'ovr', 'class_weight':'balanced', 'penalty':'l2'}, 
                                'accuracy': accuracy_score(y_test, y_pred), 
                                'precision': precision_score(y_test, y_pred, average='macro'), 
                                'recall': recall_score(y_test, y_pred, average='macro'), 
                                'F1': f1_score(y_test, y_pred, average="macro")}, ignore_index=True)


clf = LinearSVC(penalty='l2', multi_class='crammer_singer', class_weight='balanced', random_state=42)
clf.fit(X_train_vec, y_train_vec)
pickle.dump(clf, open("svm_model_2.sav", 'wb'))

y_pred = clf.predict(X_test_vec)
model_perf = model_perf.append({'ngram':'uni',
                                'model': 'svm', 
                                'parameters': {'multi_class':'crammer_singer', 'class_weight':'balanced', 'penalty':'l2'}, 
                                'accuracy': accuracy_score(y_test, y_pred), 
                                'precision': precision_score(y_test, y_pred, average='macro'), 
                                'recall': recall_score(y_test, y_pred, average='macro'), 
                                'F1': f1_score(y_test, y_pred, average="macro")}, ignore_index=True)


clf = LinearSVC(random_state=42)
clf.fit(X_train_vec, y_train_vec)
pickle.dump(clf, open("svm_model_3.sav", 'wb'))

y_pred = clf.predict(X_test_vec)
model_perf = model_perf.append({'ngram':'uni',
                                'model': 'svm', 
                                'parameters': {'multi_class':'ovr', 'class_weight':'None', 'penalty':'l2'}, 
                                'accuracy': accuracy_score(y_test, y_pred), 
                                'precision': precision_score(y_test, y_pred, average='macro'), 
                                'recall': recall_score(y_test, y_pred, average='macro'), 
                                'F1': f1_score(y_test, y_pred, average="macro")}, ignore_index=True)


"""Unigrams + Bigrams"""
#Creating the features (tf-idf weights) for the processed text
texts = X_train.astype('str')

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                   min_df = 5, 
                                   max_df = .5,
                                   max_features = 100000)
X_train_vec = tfidf_vectorizer.fit_transform(texts) #features
y_train_vec = y_train.values #target

print(f"Dimensions of X_train with unigrams and bigrams = {X_train_vec.shape}", file=f)
print(f"Dimensions of y_train = {y_train_vec.shape}", file=f)
pickle.dump(tfidf_vectorizer, open("bigram_vectorizer.pickle", "wb"))

clf = LinearSVC(penalty='l2', multi_class='crammer_singer', class_weight='balanced', random_state=42)
clf.fit(X_train_vec, y_train_vec)
pickle.dump(clf, open("svm_model_4.sav", 'wb'))

X_test_vec = tfidf_vectorizer.transform(X_test.astype('str'))
y_pred = clf.predict(X_test_vec)
model_perf = model_perf.append({'ngram':'uni + bi',
                                'model': 'svm', 
                                'parameters': {'multi_class':'crammer_singer', 'class_weight':'balanced', 'penalty':'l2'}, 
                                'accuracy': accuracy_score(y_test, y_pred), 
                                'precision': precision_score(y_test, y_pred, average='macro'), 
                                'recall': recall_score(y_test, y_pred, average='macro'), 
                                'F1': f1_score(y_test, y_pred, average="macro")}, ignore_index=True)

print(f"\nModel performance comparison:\n{model_perf}", file=f)

f.close()