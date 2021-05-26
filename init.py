from functions.data_process import prepare_dataset
from functions.data_process import cleaning_data
from xml.dom import minidom

def Init():
	config = minidom.parse('start.xml')
	mode = config.getElementsByTagName('mode')[0].attributes["value"].value
	filesTag=config.getElementsByTagName('files')[0]
	fakeCSV = filesTag.getElementsByTagName('fake')[0].attributes["value"].value
	trueCSV = filesTag.getElementsByTagName('true')[0].attributes["value"].value
	labelsTag = config.getElementsByTagName('labels')[0]
	fakeLabel = labelsTag.getElementsByTagName('fake')[0].attributes["value"].value
	trueLabel = labelsTag.getElementsByTagName('true')[0].attributes["value"].value
	verificationLabel = labelsTag.getElementsByTagName('verification')[0].attributes["value"].value
	percentage = float(config.getElementsByTagName('percentage')[0].attributes["value"].value)
	type = config.getElementsByTagName('type')[0].attributes["value"].value
	saveModel = config.getElementsByTagName('saveModel')[0].attributes["value"].value
	dataset = prepare_dataset(fakeCSV,trueCSV,fakeLabel,trueLabel,verificationLabel,percentage)[0];
	dataset['text'] = dataset['text'].apply(lambda x : cleaning_data(x))
	return [dataset,mode,type,saveModel]