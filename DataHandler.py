import numpy as np
import requests
import csv
import pandas as pd

class DataHandler:
    fileFolder='data'
    urls = {
        'trainFeatures' : "https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv",
        'trainDecisionClasses' : "https://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv",
        'testFeatures' : "https://s3.amazonaws.com/drivendata/data/7/public/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv",
        'testDecisionClasses' : "https://s3.amazonaws.com/drivendata/data/7/public/SubmissionFormat.csv"
    }

    @staticmethod
    def downloadAndSave(fileFolder = fileFolder):
        for description, url in DataHandler.urls.items():
            response = requests.get(url, stream=True)
            data = response.iter_lines(decode_unicode='utf-8')
            reader = csv.reader(data)
            with open(str(description)+'.csv','w',newline='') as f:
                writer=csv.writer(f)
                writer.writerows(reader)
    
    @staticmethod
    def readData(fileFolder=fileFolder):
        trainFeatures = pd.read_csv(fileFolder+'/'+'trainFeatures.csv')
        trainDecisionClasses = pd.read_csv(fileFolder+'/'+'trainDecisionClasses.csv')
        testFeatures = pd.read_csv(fileFolder+'/'+'testFeatures.csv')
        testDecisionClasses = pd.read_csv(fileFolder+'/'+'testDecisionClasses.csv')
        return {'trainFeatures' : trainFeatures,
                'trainDecisionClasses': trainDecisionClasses,
                'testFeatures': testFeatures,
                'testDecisionClasses': testDecisionClasses}

    @staticmethod
    def saveData(datasets, fileFolder=fileFolder):
        for name, dataset in datasets.items():
            dataset.to_csv(fileFolder+'/'+name+'.csv', sep=",", index = False)




  