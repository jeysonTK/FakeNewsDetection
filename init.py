from xml.dom import minidom

def Init():
	print("Init data...")
	print ( "Parsing config file..." )
	config = minidom.parse('start.xml')
	mode = config.getElementsByTagName('mode')[0].attributes["value"].value
	print ( "Start in ["+mode+"]" )
	filesTag=config.getElementsByTagName('files')[0]
	fakeCSV = filesTag.getElementsByTagName('fake')[0].attributes["value"].value
	print ( "File with fake news ["+fakeCSV+"]" )
	trueCSV = filesTag.getElementsByTagName('true')[0].attributes["value"].value
	print ( "File with fake news ["+trueCSV+"]" )
	labelsTag = config.getElementsByTagName('labels')[0]
	fakeLabel = labelsTag.getElementsByTagName('fake')[0].attributes["value"].value
	trueLabel = labelsTag.getElementsByTagName('true')[0].attributes["value"].value
	verificationLabel = labelsTag.getElementsByTagName('verification')[0].attributes["value"].value
	percentage = float(config.getElementsByTagName('percentage')[0].attributes["value"].value)
	type = config.getElementsByTagName('type')[0].attributes["value"].value
	print ( "Type ["+type+"]" )
	saveProc = config.getElementsByTagName('saveProc')[0].attributes["value"].value
	print ( "Save processed filename ["+saveProc+"]" )
	clean = config.getElementsByTagName('clean')[0].attributes["value"].value
	print ( "Clean ["+clean+"]" )
	cleanLevel = config.getElementsByTagName('cleanLevel')[0].attributes["value"].value
	print ( "Clean level["+cleanLevel+"]" )
	saveModel = config.getElementsByTagName('saveModel')[0].attributes["value"].value
	print ("End of init data...")
	
	return [mode,type,saveModel,fakeCSV,trueCSV,fakeLabel,trueLabel,verificationLabel,percentage,clean,saveProc,cleanLevel];
	