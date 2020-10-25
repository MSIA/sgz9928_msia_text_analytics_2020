import pandas as pd
import pickle
from string import punctuation

def process_text(text):
    # apply a function that strips punctuation and converts to lower case.
    tokens = map(lambda x: x.strip(punctuation).lower(), text.split()) 
    tokens = list(filter(None, tokens))
    return tokens

def svm_predict(text):
    new_df = pd.DataFrame({'texts':text}, index=[0])

    #preprocessing new text
    new_df['clean_text'] = new_df['texts'].apply(process_text)
    X = new_df['clean_text']

    #loading trained tfidf model and transforming new text
    tfidf = pickle.load(open("bigram_vectorizer.pickle", 'rb'))
    X_test = tfidf.transform(X.astype('str'))

    #loading trained SVM model
    clf = pickle.load(open("svm_model_4.sav", 'rb'))

    #predicting label of new text
    y_pred = clf.predict(X_test)
    label_num_dict = {0:'1Star', 1:'2Stars', 2:'3Stars', 3:'4Stars', 4:'5Stars'}
    y_label = label_num_dict.get(y_pred[0])

    return {'label': y_label}

if __name__=="__main__":

    review1 = "The food was Absolutely delicious and the service was great!! I enjoyed the ambiance on date night with bae. We sat in the lounge and the booths were cute and comfortable! The soft music was an extra added touch bc it wasn't too loud that you couldn't enjoy conversation but it was also very relaxing. They also have a TV in the lounge that they were more than happy to change to the basketball game. \nDrinks: Bae had a Tito's and Tonic and I had a mimosa. Definitely no complaints. They do have a few drink specials throughout the week. (FYI - They have a drink on the menu called Cardi B which I think is super cute).\nFood: Appetizer - Sesame Chicken - it was absolutely delicious. It also comes with honey mustard which was a great added touch. We both enjoyed them. \nI had two sushi rolls, with cream cheese, cucumbers, tempura shrimp and crab salad. Bae had chicken and boiled rice. He didn't have any complaints he too enjoyed his meal. Will definitely come back to visit as a go to sushi place this summer!"
    prediction = svm_predict(review1)
    print(f"Predicted label for new review 1 = {prediction['label']}")

    review2 = "This is my fave Thai yet! Excellent flavors, prepared perfectly, and very nice staff!! We had dumplings, pad Thai, crispy chicken, and yellow curry. Perfect! Looking forward to going back very soon. They took great care of us. Thank you!"
    prediction = svm_predict(review2)
    print(f"Predicted label for new review 2 = {prediction['label']}")

    review3 = "Allie was a great server. The service really contributed to the great atmosphere and great food. Overall great experience"
    prediction = svm_predict(review3)
    print(f"Predicted label for new review 3 = {prediction['label']}")
