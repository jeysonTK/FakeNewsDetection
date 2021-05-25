import re
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')

def prepare_dataset(fakeCsv,trueCsv,fakeLabel,trueLabel,testHead,train_percentage): 
	#Read data from files
	fake = pd.read_csv(fakeCsv)
	true = pd.read_csv(trueCsv)
	
	#Label
	fake["label"]=fakeLabel
	true["label"]=trueLabel
	
	#remove missing values
	fake.dropna()
	true.dropna()
	
	#Calculate size for train and test data
	fake_percentage = int(fake[testHead].size*train_percentage)
	true_percentage = int(true[testHead].size*train_percentage)

	#Separate test data from train data
	train_fake_data = fake[0:fake_percentage]
	train_true_data = true[0:true_percentage]
	test_fake_data = fake[fake_percentage:fake[testHead].size]
	test_true_data = true[true_percentage:true[testHead].size]
	
	# Concating:
	data_train = pd.concat([train_true_data, train_fake_data], axis=0)
	data_test = pd.concat([test_true_data, test_fake_data], axis=0)
	
	data_train = data_train.sample(frac = 1)
	data_test = data_test.sample(frac = 1)
	
	#Shuffle
	data_train = data_train.sample(frac = 1)
	data_test = data_test.sample(frac = 1)
	
	dataset = [data_train[[testHead,'label']], data_test[[testHead,'label']]]
	return dataset

def load_prepared_dataset(trainCsv,testCsv,testHead):
	#Reload data from files
	data_train = pd.read_csv(trainCsv)
	data_test = pd.read_csv(testCsv)

	#Shuffle
	data_train = data_train.sample(frac = 1)
	data_test = data_test.sample(frac = 1)
	
	dataset = [data_train[[testHead,'label']], data_test[[testHead,'label']]]
	return dataset
	
def cleaning_data(row):
	
	ps = WordNetLemmatizer()
	stopwords1 = stopwords.words('english')
	
	# convert text to into lower case
	row = row.lower()
	
	# this line of code only take words from text and remove number and special character using RegX
	row = re.sub('[^a-zA-Z]' , ' ' , row)
	
	# split the data and make token.
	token = row.split() 
	
	# lemmatize the word and remove stop words like a, an , the , is ,are ...
	news = [ps.lemmatize(word) for word in token if not word in stopwords1]  
	
	# finaly join all the token with space
	cleanned_news = ' '.join(news) 
	
	# return cleanned data
	return cleanned_news 
