import sys
from init import Init
from functions.data_process import prepare_dataset
from functions.model_process import predict_ml
from functions.model_process import train_ml
from functions.model_process import single_predict_ml

#init returns: 	0 = MODE; 		1 = TYPE; 			2 = Filename for save model; 	3 = Fake CSV; 
# 				4 = True CSV	5 = Label for fake; 6 = Label for true; 			7 = verification label; 		
#				8 = percentage	9 = clean
init_data = Init()

def switch(argument):
	if argument == "TEST_ML":
		predict_ml(dataset, init_data[2])
	elif argument == "TRAIN_ML":
		print ( "Preparing dataset..." )
		dataset = prepare_dataset(init_data[3],init_data[4],init_data[5],init_data[6],init_data[7],init_data[8],init_data[9]);
		train_ml(dataset, init_data[2])
	elif argument == "STEST_ML":
		single_predict_ml(str(sys.argv[1]),init_data[2],init_data[9])
switch( (init_data[0]+"_"+init_data[1]))

