import sys
from init import Init
from functions.model_process import predict_ml

init_data = Init()
dataset = init_data[0]

print (dataset)
def switch(argument):
	switcher = {
		"TEST_ML": predict_ml(dataset)
	}

switch( (init_data[1]+"_"+init_data[2]))

