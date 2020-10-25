# HW-2

**Name:** Ganguly, Shreyashi

**NetID:** sgz9928

GitHub link - https://github.com/MSIA/sgz9928_msia_text_analytics_2020.git<br>
Branch - homework3

## Problem 1

Download and familiarize yourself with a large text classification corpus.

Yelp review dataset was chosen as the text corpus.

The data was in the form of a json file with each review having the following elements,
- review_id
- user_id
- business_id
- stars (multiclass label for the text classification task)
- useful, funny, cool (multi labels)
- text (actual free text review)
- date

Sample review from the corpus:
{"review_id":"xQY8N_XvtGbearJ5X4QryQ",<br>"user_id":"OwjRMXRC0KyPrIlcjaXeFQ",<br>"business_id":"-MhfebM0QIsKt87iDN-FNw",<br>
"stars":2.0,<br>
"useful":5,"funny":0,"cool":0,<br>
"text":"As someone who has worked with many museums, I was eager to visit this gallery on my most recent trip to Las Vegas. When I saw they would be showing infamous eggs of the House of Faberge from the Virginia Museum of Fine Arts (VMFA), I knew I had to go!\n\nTucked away near the gelateria and the garden, the Gallery is pretty much hidden from view. It's what real estate agents would call \"cozy\" or \"charming\" - basically any euphemism for small.\n\nThat being said, you can still see wonderful art at a gallery of any size, so why the two *s you ask? Let me tell you:\n\n* pricing for this, while relatively inexpensive for a Las Vegas attraction, is completely over the top. For the space and the amount of art you can fit in there, it is a bit much.\n* it's not kid friendly at all. Seriously, don't bring them.\n* the security is not trained properly for the show. When the curating and design teams collaborate for exhibitions, there is a definite flow. That means visitors should view the art in a certain sequence, whether it be by historical period or cultural significance (this is how audio guides are usually developed). When I arrived in the gallery I could not tell where to start, and security was certainly not helpful. I was told to \"just look around\" and \"do whatever.\" \n\nAt such a *fine* institution, I find the lack of knowledge and respect for the art appalling.",<br>"date":"2015-04-15 05:21:16"}


Number of documents in corpus = 500000<br>
Unique labels = {5: 220375, 4: 112802, 1: 70468, 3: 55778, 2: 40577}<br>

Distribution of labels:<br>
{1: 0.140936, 2:0.081154, 3:0.111556, 4:0.225604, 5:0.44075}

Average word length of documents = 106.974186

The relevant script for this problem - [here]() <br>
 


## Problem 2

Build a logistic regression text classifier.

Logistic Regression classifier was built using the scikit learn library

The following steps were followed:
- Extracted the text review and stars from the dataset
- Tokenized the text, removed punctuations and converted to lower case
- Encoded the labels
- Split the data into training and test sets in the ratio 80:20. Used stratified sampling to ensure equal distribution of target labels in training and test data
- Used TfidfVectorizer from scikit learn to convert words to vectors. Excluded words that appeared in more that 50% of the documents or in less than 5 documents. Tfidf model was trained on training alone
- Tried using unigram and combination of uni and bigrams
- Experimented with 'multiclass', 'penalty', 'solver' and 'classweight' hyperparameters
- Selected the best model based on its performance on the test data

Please find below the results of the experimentation:

Distribution of labels in trianing data:<br>
[0 1 2 3 4] [0.140935 0.081155 0.111555 0.225605 0.44075 ]

Distribution of labels in test data:<br>
[0 1 2 3 4] [0.14094 0.08115 0.11156 0.2256  0.44075]

Dimensions of X_train with unigrams = (400000, 44474) <br>
Dimensions of y_train = (400000,)

Dimensions of X_train with unigrams and bigrams = (400000, 100000)<br>
Dimensions of y_train = (400000,)
<br>
<br>

Model performance comparison:

|  ngram   |  model   |  parameters  |  accuracy  |  precision  |  recall  |  F1  |
| --- | --- | --- | --- | --- | --- | --- |
|  uni   |  logistic regression  |  {'multi_class':'ovr', 'class_weight':'balanced', 'penalty':'l2'}  |  0.66766 |  0.586625 |  0.594742 |  0.589955 |
|  uni   |  logistic regression  |  {'multi_class':'ovr', 'class_weight':'balanced', 'penalty':'l1'}  |  0.66706 |  0.585104 |  0.594165 |  0.588941 |
|  uni   |  logistic regression  |  {'multi_class':'multinomial', 'class_weight':'balanced', 'solver':'lbfgs', 'penalty':'l2', 'max_iter':200}  |  0.64742 |  0.583991 |  0.605696 |  0.592081 |
|  uni+bi   |  logistic regression  |  {'multi_class':'ovr', 'class_weight':'balanced', 'solver':'liblinear', 'penalty':'l2'}  |  0.69159 |  0.616493 |  0.621065 |  0.618226 |

Based on the performance on the test data the last model with both unigrams and bigrams is selected as the best model.

The relevant script for this problem - [here]() <br>


## Problem 3

Build a linear SVM text classifier.

Scikit-learn LinearSVC was used to build the SVM model.

The following steps were followed:
- Extracted the text review and stars from the dataset
- Tokenized the text, removed punctuations and converted to lower case
- Encoded the labels
- Split the data into training and test sets in the ratio 80:20. Used stratified sampling to ensure equal distribution of target labels in training and test data
- Used TfidfVectorizer from scikit learn to convert words to vectors. Excluded words that appeared in more that 50% of the documents or in less than 5 documents. Tfidf model was trained on training alone
- Tried using unigram and combination of uni and bigrams
- Experimented with 'multiclass', 'penalty' and 'classweight' hyperparameters
- Selected the best model based on its performance on the test data

The vector size etc. are the same as displayed in Problem 2

Model performance comparison:

|  ngram   |  model   |  parameters  |  accuracy  |  precision  |  recall  |  F1  |
| --- | --- | --- | --- | --- | --- | --- |
|  uni   |  svm  |  {'multi_class':'ovr', 'class_weight':'balanced', 'penalty':'l2'}  |  0.64929 |  0.562065 |  0.573579 |  0.566755 |
|  uni   |  svm  |  {'multi_class':'crammer_singer', 'class_weight':'balanced', 'penalty':'l2'}  |  0.64050 |  0.564958 |  0.583244 |  0.572410 |
|  uni   |  svm  |  {'multi_class':'ovr', 'class_weight':'None', 'penalty':'l2'}  |  0.66045 |  0.575347 |  0.546057 |  0.550908 |
|  uni+bi   |  svm  |  {'multi_class':'crammer_singer', 'class_weight':'balanced', 'penalty':'l2'}  |  0.65474 |  0.581460 |  0.596608 |  0.587884 |


Though the accuracy of the uni+bigram model is marginally lower, it has the highest F1-score. Considering the unbalanced nature of the dataset, F1-score is more important and hence the last model (uni+bi) is selected as the best model

The relevant script for this problem - [here]() <br>


## Problem 4

Write a “predict” script for your best-performing SVM model (based on your hyperparameter tuning experiments)

review1 = "The food was Absolutely delicious and the service was great!! I enjoyed the ambiance on date night with bae. We sat in the lounge and the booths were cute and comfortable! The soft music was an extra added touch bc it wasn't too loud that you couldn't enjoy conversation but it was also very relaxing. They also have a TV in the lounge that they were more than happy to change to the basketball game. \nDrinks: Bae had a Tito's and Tonic and I had a mimosa. Definitely no complaints. They do have a few drink specials throughout the week. (FYI - They have a drink on the menu called Cardi B which I think is super cute).\nFood: Appetizer - Sesame Chicken - it was absolutely delicious. It also comes with honey mustard which was a great added touch. We both enjoyed them. \nI had two sushi rolls, with cream cheese, cucumbers, tempura shrimp and crab salad. Bae had chicken and boiled rice. He didn't have any complaints he too enjoyed his meal. Will definitely come back to visit as a go to sushi place this summer!"

Predicted label for new review 1 = 4Stars
<br>
<br>
review2 = "This is my fave Thai yet! Excellent flavors, prepared perfectly, and very nice staff!! We had dumplings, pad Thai, crispy chicken, and yellow curry. Perfect! Looking forward to going back very soon. They took great care of us. Thank you!"

Predicted label for new review 2 = 5Stars
<br>
<br>
review3 = "Allie was a great server. The service really contributed to the great atmosphere and great food. Overall great experience"

Predicted label for new review 3 = 4Stars

The relevant script for this problem - [here]() <br>
 





