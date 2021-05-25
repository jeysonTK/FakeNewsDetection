from functions.data_process import prepare_dataset
from functions.data_process import cleaning_data

dataset = prepare_dataset("Fake.csv","True.csv","FAKE","REAL","text",1)[0];

dataset['text'] = dataset['text'].apply(lambda x : cleaning_data(x))

print(dataset);