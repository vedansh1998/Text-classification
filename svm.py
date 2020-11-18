import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
# This is used to reproduce the same result every time if the script is kept consistent otherwise each run will produce different results.
np.random.seed(500)

# The data set can be easily added as a pandas Data Frame with the help of ‘read_csv’ function.
Corpus = pd.read_csv(r"./input/tex.csv",encoding='latin-1')
Corpus_test = pd.read_csv(r"./Mined_data.csv",encoding='latin-1')
# print(Corpus)

# Data pre-processing
# 1 Tokenization
# 2 Word Stemming/Lemmatization


# Step - a : Remove blank rows if any.
Corpus['text'].dropna(inplace=True)
# Corpus_test['text'].dropna(inplace=True)


# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Corpus['text'] = [entry.lower() for entry in Corpus['text']]
# Corpus_test['text'] = [entry.lower() for entry in Corpus_test['text']]

# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
# Corpus_test['text'] = [entry.lower() for entry in Corpus_test['text']]

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.


# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(Corpus['text']):
	# Declaring Empty List to store the words that follow the rules for this step
	Final_words = []
	# Initializing WordNetLemmatizer()
	word_Lemmatized = WordNetLemmatizer()
	# pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
	for word, tag in pos_tag(entry):
		# Below condition is to check for Stop words and consider only alphabets
		if word not in stopwords.words('english') and word.isalpha():
			word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
			Final_words.append(word_Final)
	# The final processed set of words for each iteration will be stored in 'text_final'
	Corpus.loc[index,'text_final'] = str(Final_words)

# for index,entry in enumerate(Corpus_test['text']):
# 	# Declaring Empty List to store the words that follow the rules for this step
# 	Final_words = []
# 	# Initializing WordNetLemmatizer()
# 	word_Lemmatized = WordNetLemmatizer()
# 	# pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
# 	for word, tag in pos_tag(entry):
# 		# Below condition is to check for Stop words and consider only alphabets
# 		if word not in stopwords.words('english') and word.isalpha():
# 			word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
# 			Final_words.append(word_Final)
# 	# The final processed set of words for each iteration will be stored in 'text_final'
# 	Corpus_test.loc[index,'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.05)

# Label encode the target variable — This is done to transform Categorical data of string type in the data set into numerical values 
# not needed in our case already in numerical format--labels
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


# Word Vectorization====TfIdf
Tfidf_vect = TfidfVectorizer(max_features=80000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
# Test_X_Tfidf = Tfidf_vect.transform(Corpus_test['text_final'])




# output --{‘even’: 1459, ‘sound’: 4067, ‘track’: 4494, ‘beautiful’: 346, ‘paint’: 3045, ‘mind’: 2740, ‘well’: 4864,} 

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# for z in predictions_SVM:
# 	print(z)

# # Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


# random forest
clf = RandomForestClassifier()
clf.fit(Train_X_Tfidf,Train_Y)
predictions_Randomforest = clf.predict(Test_X_Tfidf)
# print(predictions_Randomforest)
# Use accuracy_score function to get the accuracy
print("Random Forest Accuracy Score -> ",accuracy_score(predictions_Randomforest,Test_Y)*100)


# # k-means
# from sklearn import cluster
# k_means = cluster.KMeans(n_clusters=2, n_init=1)
# k_means.fit(X_t)

# logistic regression
# classifier
classifier = LogisticRegression()
classifier.fit(Train_X_Tfidf, Train_Y)
predictions_logistic = classifier.predict(Test_X_Tfidf)
# print(predictions_logistic)
print("logistic Accuracy Score -> ",accuracy_score(predictions_logistic,Test_Y)*100)


# print(predictions_Randomforest)

# import json
# t=0
# import csv  
# with open('bodyshaming.json') as json_file:
# 	data = json.load(json_file)
# 	for d in data:
# 		books = d['edge_media_to_caption']['edges'][0]['node']['text']
# 		# print(books)
