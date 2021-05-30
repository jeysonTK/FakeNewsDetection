import re
import nltk
import pylab as pl
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def prepare_dataset(fakeCsv,trueCsv,fakeLabel,trueLabel,testHead,train_percentage, clean): 
	print ( "Read data from files..." )
	fake = pd.read_csv(fakeCsv)
	true = pd.read_csv(trueCsv)
	
	print ( "Labeling data..." )
	fake["label"]=fakeLabel
	true["label"]=trueLabel
	
	print ( "Remove missing values..." )
	fake.dropna()
	true.dropna()
	
	print ( "Calculate size for train and test data.." )
	fake_percentage = int(fake[testHead].size*train_percentage)
	true_percentage = int(true[testHead].size*train_percentage)

	print ( "Separate test data from train data..." )
	train_fake_data = fake[0:fake_percentage]
	train_true_data = true[0:true_percentage]
	test_fake_data = fake[fake_percentage:fake[testHead].size]
	test_true_data = true[true_percentage:true[testHead].size]
	

	print ( "Concating..." )
	data_train = pd.concat([train_true_data, train_fake_data], axis=0)
	data_test = pd.concat([test_true_data, test_fake_data], axis=0)
	
	data_train = data_train.sample(frac = 1)
	data_test = data_test.sample(frac = 1)
	
	#Shuffle
	data_train = data_train.sample(frac = 1)
	data_test = data_test.sample(frac = 1)
	
	dataset = [data_train[[testHead,'label']], data_test[[testHead,'label']]]
	dataset = dataset[0]
	if clean.lower() == "true":
		print ( "Cleaning data...")
		dataset[testHead] = dataset[testHead].apply(lambda x : cleaning_data(x))
	
	print ( dataset )
	return dataset
	
def dataset_statistics(fakeCsv,trueCsv,fakeLabel,trueLabel,testHead,train_percentage, clean,saveProc,cleanLevel): 
	print ( "Generate statistics..." )
	print ( "Read data from files..." )
	fake = pd.read_csv(fakeCsv)
	true = pd.read_csv(trueCsv)
	
	train_percentage = int(train_percentage)
	if train_percentage < 5:
		train_percentage = 5
	
	to_save_com = ""
	
	clean_level = int(cleanLevel)
	
	if clean.lower() == "true":
		to_save_com = "cleaned"
		if clean_level == 1:
			print ( "Cleaning data L1...")
			fake[testHead] = fake[testHead].apply(lambda x : soft_cleaning_data(x))
			true[testHead] = true[testHead].apply(lambda x : soft_cleaning_data(x))
			to_save_com = "cleaned_Ll"
		if clean_level == 2:
			print ( "Cleaning data L2...")
			fake[testHead] = fake[testHead].apply(lambda x : cleaning_data(x))
			true[testHead] = true[testHead].apply(lambda x : cleaning_data(x))
			to_save_com = "cleaned_L2"
		if clean_level == 3:
			print ( "Cleaning data L3...")
			fake[testHead] = fake[testHead].apply(lambda x : xtream_cleaning_data(x))
			true[testHead] = true[testHead].apply(lambda x : xtream_cleaning_data(x))
			to_save_com = "cleaned_L3"
	else:
		to_save_com = "not_cleaned"
	
	print ( "Make word dictionary from fake news..." )
	fake_dict = word_dict(fake[testHead])
	print ( "Make word dictionary from real news..." )
	true_dict = word_dict(true[testHead])
	
	print ( "Search for word in news that is in fake and in real..." )
	dict_fake_and_true = union_dict(fake_dict,true_dict)
	print ( "Search for word in news that is in real and in fake..." )
	dict_true_and_fake = union_dict(true_dict,fake_dict)
	
	print ( "Search for word in news that is in fake and NOT in real..." )
	dict_fake_not_in_true = not_in(fake_dict,true_dict)
	print ( "Search for word in news that is in real and NOT in fake..." )
	dict_true_not_in_fake = not_in(true_dict,fake_dict)

	print ( "Convert obtained dictionary to CSV" )
	
	if bool(fake_dict) ==  True:
		fake_word_count_CSV = pd.DataFrame(fake_dict.items()).sort_values(by=1,ascending=False)
	if bool(true_dict) ==  True:
		true_word_count_CSV = pd.DataFrame(true_dict.items()).sort_values(by=1,ascending=False)
	if bool(dict_fake_and_true) ==  True:
		dict_fake_and_true_CSV=pd.DataFrame(dict_fake_and_true.items()).sort_values(by=1,ascending=False)
	if bool(dict_true_and_fake) ==  True:
		dict_true_and_fake_CSV=pd.DataFrame(dict_true_and_fake.items()).sort_values(by=1,ascending=False)
	if bool(dict_fake_not_in_true) ==  True:
		dict_fake_not_in_true_CSV=pd.DataFrame(dict_fake_not_in_true.items()).sort_values(by=1,ascending=False)
	if bool(dict_true_not_in_fake) ==  True:
		dict_true_not_in_fake_CSV=pd.DataFrame(dict_true_not_in_fake.items()).sort_values(by=1,ascending=False)

	if saveProc != "":
		print( "Save processed data" )
		with open( "fake_"+saveProc+to_save_com, 'w') as f:
			f.write("word,no\n")
			for key in fake_dict.keys():
				f.write("%s,%s\n"%(key,fake_dict[key]))
		with open( "true_"+saveProc+to_save_com, 'w') as f:
			f.write("word,no\n")
			for key in true_dict.keys():
				f.write("%s,%s\n"%(key,true_dict[key]))
				
	print ( "Increase figure size" )			
	pl.rcParams["figure.figsize"] = (15, 5)
	
	print ( "Save figures" )	
	if bool(fake_dict) ==  True:
		fake_myplot = fake_word_count_CSV[0:train_percentage].plot.bar(x=0, y=1, rot=0)
		fake_myplot.figure.savefig("false_"+saveProc+to_save_com+".pdf")
		
	if bool(true_dict) ==  True:
		true_myplot = true_word_count_CSV[0:train_percentage].plot.bar(x=0, y=1, rot=0)
		true_myplot.figure.savefig("true_"+saveProc+to_save_com+".pdf")
		
	if bool(dict_fake_and_true) ==  True:
		dict_fake_and_true_CSV_myplot = dict_fake_and_true_CSV[0:train_percentage].plot.bar(x=0, y=1, rot=0)
		dict_fake_and_true_CSV_myplot.figure.savefig("Fake_And_True_"+saveProc+to_save_com+".pdf")

		dict_fake_and_true_CSV_myplot2 = dict_fake_and_true_CSV[(len(dict_fake_and_true_CSV)-train_percentage):len(dict_fake_and_true_CSV)].plot.bar(x=0, y=1, rot=0)
		dict_fake_and_true_CSV_myplot2.figure.savefig("Fake_And_True_Min_"+saveProc+to_save_com+".pdf")
		
	if bool(dict_true_and_fake) ==  True:
		dict_true_and_fake_CSV_myplot = dict_true_and_fake_CSV[0:train_percentage].plot.bar(x=0, y=1, rot=0)
		dict_true_and_fake_CSV_myplot.figure.savefig("True_And_Fake_"+saveProc+to_save_com+".pdf")

		dict_true_and_fake_CSV_myplot2 = dict_true_and_fake_CSV[(len(dict_true_and_fake_CSV)-train_percentage):len(dict_true_and_fake_CSV)].plot.bar(x=0, y=1, rot=0)
		dict_true_and_fake_CSV_myplot2.figure.savefig("True_And_Fake_Min."+saveProc+to_save_com+".pdf")
		
	if bool(dict_fake_not_in_true) ==  True:
		dict_fake_not_in_true_CSV_myplot = dict_fake_not_in_true_CSV[0:train_percentage].plot.bar(x=0, y=1, rot=0)
		dict_fake_not_in_true_CSV_myplot.figure.savefig("fake_not_in_true_"+saveProc+to_save_com+".pdf")
		
	if bool(dict_true_not_in_fake) ==  True:
		dict_true_not_in_fake_CSV_myplot = dict_true_not_in_fake_CSV[0:train_percentage].plot.bar(x=0, y=1, rot=0)
		dict_true_not_in_fake_CSV_myplot.figure.savefig("true_not_in_fake_"+saveProc+to_save_com+".pdf")

	
def load_prepared_dataset(trainCsv,testCsv,testHead):
	#Reload data from files
	data_train = pd.read_csv(trainCsv)
	data_test = pd.read_csv(testCsv)

	#Shuffle
	data_train = data_train.sample(frac = 1)
	data_test = data_test.sample(frac = 1)
	
	dataset = [data_train[[testHead,'label']], data_test[[testHead,'label']]]
	return dataset
	
def xtream_cleaning_data(row):
	ps = WordNetLemmatizer()
	stopwords1 = stopwords.words('english')
	stopwords1.extend(["monday","tuesday","wednesday","thursday","friday","saturday","sunday"])
	stopwords1.extend(["january","february","march","april","may","june","july","august","september","october ","november","december"])
	
	row = row.replace("U.S.", "UnitedStates")
	row = row.lower()
	row = re.sub('[^a-zA-Z]' , ' ' , row)
	
	token = row.split() 
	
	news = [ps.lemmatize(word) for word in token if not word in stopwords1]  
	cleanned_news = ' '.join(news) 
	
	return cleanned_news 
	
def cleaning_data(row):
	ps = WordNetLemmatizer()
	stopwords1 = stopwords.words('english')
	
	row = row.lower()
	row = re.sub('[^a-zA-Z]' , ' ' , row)
	
	token = row.split() 
	
	news = [ps.lemmatize(word) for word in token if not word in stopwords1]  
	cleanned_news = ' '.join(news) 
	
	return cleanned_news 
	
def soft_cleaning_data(row):
	ps = WordNetLemmatizer()
	stopwords1 = stopwords.words('english')
	row = re.sub('[^a-zA-Z]' , ' ' , row)
	
	token = row.split() 
	
	news = [ps.lemmatize(word) for word in token if not word.lower() in stopwords1]  
	cleanned_news = ' '.join(news) 
	
	return cleanned_news 
	
def word_dict(column):
	dict = {}
	
	for x in column:
		funique =set(" ".join(str(x).split()).split(" "))
		for word in funique:
			if word in dict :
				dict[word] += 1
			else:
				dict[word] = 1
	return dict

def union_dict(dict1,dict2):
	dict3 = {}
	for key in dict1.keys():
		if key in dict2.keys() :
			dict3[key] = dict1[key] - dict2[key]
	
	return dict3

def not_in(dict1,dict2):
	dict3 = {}
	for key in dict1.keys():
		if key not in dict2.keys() :
			dict3[key] = dict1[key]
			
	return dict3