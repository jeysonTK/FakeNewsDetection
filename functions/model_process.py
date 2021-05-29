import joblib 
import pickle
import numpy as np
import pandas as pd
from init import Init
from .data_process import cleaning_data
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report

def predict_ml(dataset, filename):
	print ("Predict ML......")
	dataset['text'] = dataset['text'].apply(lambda x : cleaning_data(x))
	vectorizer = TfidfVectorizer(max_features = 50000 , lowercase=False , ngram_range=(1,2))
	
	X = dataset.iloc[:35000,0]
	y = dataset.iloc[:35000,1]
	
	train_data , test_data , train_label , test_label = train_test_split(X , y , test_size = 1,random_state = 0)
	
	train_data.shape , test_data.shape
	vec_test_data = vectorizer.transform(test_data).toarray()
	vec_test_data.shape
	test_label.value_counts() # balanced partition
	
	testing_data = pd.DataFrame(vec_test_data , columns= vectorizer.get_feature_names())
	
	model = joblib.load(filename)
	y_pred  = model.predict(testing_data)
	
	pd.Series(y_pred).value_counts()
	test_label.value_counts()
	print(classification_report(test_label , y_pred))
	
	accuracy_score(test_label , y_pred)
	print("END predict ML...")
	
def train_ml(dataset, filename):
	print ("Train ML...")
	print("Vectorize data...")
	vectorizer = TfidfVectorizer(max_features = 50000 , lowercase=False , ngram_range=(1,2))
	
	X = dataset.iloc[:35000,0]
	y = dataset.iloc[:35000,1]
	print( "Split data..." )
	train_data , test_data , train_label , test_label = train_test_split(X , y , test_size = 0.2 ,random_state = 0)

	vec_train_data = vectorizer.fit_transform(train_data)
	vec_train_data = vec_train_data.toarray()
	train_data.shape , test_data.shape
	vec_test_data = vectorizer.transform(test_data).toarray()
	vec_train_data.shape , vec_test_data.shape
	train_label.value_counts() # balanced partition
	test_label.value_counts() # balanced partition

	print ("To dataframe")
	training_data = pd.DataFrame(vec_train_data , columns=vectorizer.get_feature_names())
	testing_data = pd.DataFrame(vec_test_data , columns= vectorizer.get_feature_names())
	print ( "Crate MultinomialNB model...." )
	clf = MultinomialNB()
	print ( "Fit MultinomialNB model...." )
	clf.fit(training_data, train_label)
	y_pred  = clf.predict(testing_data)
	print ( "Predict test data with MultinomialNB model...." )
	y_pred_train = clf.predict(training_data)
	print(classification_report(train_label , y_pred_train))
	
	accuracy_score(train_label , y_pred_train)
	accuracy_score(test_label , y_pred)
	
	with open(filename, 'wb') as fout:
		pickle.dump((vectorizer, clf), fout)
	
	
def single_predict_ml(news,filename,clean):
	print("Loading model ["+filename+"]")
	
	with open(filename, 'rb') as f:
		vectorizer, model = pickle.load(f)
	print (vectorizer)
	if clean.lower() == "true":
		print ( "Cleaning data...")
		news = cleaning_data(news)
		print( news )
	
	single_prediction = model.predict(vectorizer.transform([news]).toarray())
	print(single_prediction)