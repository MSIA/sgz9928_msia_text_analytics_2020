import json
import pandas as pd
from string import punctuation
from sklearn.preprocessing import LabelEncoder
import fasttext

file="yelp_dataset/yelp_academic_dataset_review.json"

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

#Turning the labels into numbers
LE = LabelEncoder()
df['stars_num'] = LE.fit_transform(df['stars'])

print(df.head())

# transforming input data as per fast text
col = ['stars_num', 'review']
yelp_reviews = df[col]
yelp_reviews['stars_num']=['__label__'+ str(s) for s in yelp_reviews['stars_num']]
yelp_reviews['review']= yelp_reviews['review'].str.strip(punctuation).replace('\n',' ', regex=True).replace('\t',' ', regex=True)
print(yelp_reviews.head())

# split into train and test
train = yelp_reviews.iloc[:400001]
train.to_csv(r'yelp_reviews_train.txt', header=None, index=None, sep=' ', mode='a')
test = yelp_reviews.iloc[400001:]
test.to_csv(r'yelp_reviews_test.txt', header=None, index=None, sep=' ', mode='a')

model = fasttext.train_supervised(input="yelp_reviews_train.txt")

print(model.test("yelp_reviews_test.txt"))




