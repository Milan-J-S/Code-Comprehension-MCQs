import json 

def parse():
    with open('output1.json') as f:
        fread = f.read()
        print(fread)
        jsondata = json.loads(fread)
        print(jsondata)
        for key, value in getAll(jsondata):
            print(key , "----->", value)
        
        

def getAll(jsondata):
    for key, value in jsondata.items():
        print(key, value)
        if type(value) is dict:
            yield(key, value)
            yield from getAll(value)
        elif(type(value) is list and len(value)>0):
            for item in value:
                if(type(item) is not str):
                    yield from getAll(item)

parse()