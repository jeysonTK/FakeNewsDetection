import joblib 
import numpy as np
import pandas as pd
from init import Init
from .data_process import cleaning_data
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report

def predict_ml(dataset):
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
	
	model = joblib.load('model.pkl')
	y_pred  = model.predict(testing_data)
	
	pd.Series(y_pred).value_counts()
	test_label.value_counts()
	print(classification_report(test_label , y_pred))
	
	accuracy_score(test_label , y_pred)