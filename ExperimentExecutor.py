import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
import pylab

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import preprocessing

from FeaturesPreprocessorHelper import FeaturesPreprocessorHelper as FPH
from DataHandler import DataHandler



class ExperimentExecutor:
    def __init__(self):
        self.trainFeatures=None
        self.trainDecisionClasses=None
        self.testFeatures=None
        self.testDecisionClasses=None

    def setData(self, datasets):
        self.trainFeatures=datasets['trainFeatures']
        self.trainDecisionClasses=datasets['trainDecisionClasses']
        self.testFeatures=datasets['testFeatures']
        self.testDecisionClasses=datasets['testDecisionClasses']



    # Tworzymy kolumny 'year_recorded' 'month_recorded' 'ordigal_date_recorded' z kolumny 'date_recorded'
    def monthYearOrdinalFromDate(self):
        self.trainFeatures, self.testFeatures = FPH.monthYearOrdinalFromDate(self.trainFeatures, self.testFeatures,'date_recorded', 'year_recorded', 'month_recorded', 'ordinal_date_recorded')

    # Tworzymy zmienne, kolmny fikcyjne (dummy variable) na podstawie kolumn 'year_recorded', 'month_recorded'
    # Należy wziąć pod uwagę, aby tworzyć nowe wartości, jedynie gdy zawierają się w zbiorze treningowym i testowym
    def monthYearToDummy(self):
        self.trainFeatures, self.testFeatures = FPH.toDummy(self.trainFeatures,self.testFeatures,['year_recorded','month_recorded'])
    
    # Zastępujemy puste miejsca w kolumnie ['construction_year'] wartością średnią        
    def constructionDateReplaceNulls(self):
        self.trainFeatures, self.testFeatures = FPH.replaceValueByMean(self.trainFeatures,self.testFeatures, ['construction_year'], 0)

    #Zastępujemy puste wartości w koumnie 'public_meeting' wartościami False
    def publicMeetingNullsToFalse(self):
        self.trainFeatures, self.testFeatures = FPH.replaceValueByValue(self.trainFeatures,self.testFeatures,['public_meeting'], np.nan, False)   

    # Zastępujemy puste wartości w koumnie 'permit' wartościami False
    def permitNullsToFalse(self):
        self.trainFeatures, self.testFeatures = FPH.replaceValueByValue(self.trainFeatures,self.testFeatures,['permit'], np.nan, False) 

    # Zastępujemy błedne wartości w kolumnie 'latitude' w celu dalszego przetwarzania
    def replaceWrongValuesInLatitudeByNan(self):
        self.trainFeatures, self.testFeatures = FPH.applyFunctionOnColumns(self.trainFeatures, self.testFeatures,['latitude'], lambda x: np.nan if (x>-0.5 or x<-13.0) else x) 
    
    # Zastępujemy błedne wartości w kolumnie 'longitude' w celu dalszego przetwarzania
    def replaceWrongValuesInLongitudeByNan(self):
        self.trainFeatures, self.testFeatures = FPH.applyFunctionOnColumns(self.trainFeatures, self.testFeatures,['longitude'], lambda x: np.nan if (x<28.5 or x > 41.5) else x)

    # Zastępujemy niepewne wartości w kolumnie 'population' wartością nan w celu dalszego przetwarzania
    def replaceWrongValuesInPopulationByNan(self):
        self.trainFeatures, self.testFeatures = FPH.applyFunctionOnColumns(self.trainFeatures, self.testFeatures,['population'], lambda x: np.nan if (x<2) else x)

    # Zastępujemy niepewne wartości w kolumnie 'gps_height' wartością nan w celu dalszego przetwarzania
    def replaceWrongValuesInGpsHeightByNan(self):
        self.trainFeatures, self.testFeatures = FPH.applyFunctionOnColumns(self.trainFeatures, self.testFeatures,['gps_height'], lambda x: np.nan if (x<10) else x)

    # Zastępujemy nieznane wartości w danych kolumnach wartościami średnimi, liczonymi na podstawie grupowania po wartościach atrybutów podanych kolumn bazowych
    def replaceUnknownValuesBySmartMean(self):
        self.trainFeatures, self.testFeatures = FPH.replaceColumnIndicatedValuesByMeanDependedOnOtherColumns(self.trainFeatures, self.testFeatures, 
                                                                                                        ['longitude', 'latitude', 'gps_height', 'population'], 
                                                                                                        ['subvillage', 'district_code', 'basin'])
    def removeColumns(self):
        columns = ['id', 'amount_tsh',  'num_private', 'wpt_name', 
                'recorded_by', 'subvillage', 'scheme_name', 'region', 
                'quantity', 'quality_group', 'source_type', 'payment', 
                'waterpoint_type_group',
                'extraction_type_group','date_recorded']
        self.trainFeatures, self.testFeatures = FPH.removeColumns(self.trainFeatures, self.testFeatures,columns)
    
    # Zmieniamy wartości na 'other' we wszystkich kolumnach, jeżeli te wartości występują żadko - mniej ni 100 razy
    def replaceTooRareValuesInStringColumns(self):
        columns = [column for column in self.trainFeatures.columns if type(self.trainFeatures[column].iloc[0]) == str]
        self.trainFeatures, self.testFeatures = FPH.reduceFactorLevels(self.trainFeatures, self.testFeatures, columns, 100, 'other')
    
    # Zmieniamy wszystkie nieznane wartości w kolumnach typu string na 'other'
    def replaceUnknownValuesInStringColumns(self):
        self.trainFeatures, self.testFeatures = FPH.replaceUnknownValuesInStringColumns(self.trainFeatures, self.testFeatures)

    # Utworzenie dummy z kolumn zawierających stringgi , wykorzystać gotową funkcję
    def createDummyFromStringColumns(self): 
        self.trainFeatures, self.testFeatures = FPH.createDummyFromStringColumns(self.trainFeatures, self.testFeatures)

    # Enkodowanie (do wartości numerycznych kolumn zawierających stringi)
    def encodeAllStringColumns(self):
        self.trainFeatures, self.testFeatures = FPH.encodeAllStringColumns(self.trainFeatures, self.testFeatures)

    # Standaryzacja zmiennych numerycznych
    def standarize(self):
        self.trainFeatures, self.testFeatures = FPH.replaceUnknownByMean(self.trainFeatures,self.testFeatures,['gps_height', 'latitude', 'longitude'])

    # Redukcja wymiarowości zmiennych numerycznych
    def dimensionalityReduction(self):
        self.trainFeatures, self.testFeatures = FPH.dimensionalityReductionByLinearDiscriminantAnalysis(self.trainFeatures,self.testFeatures , self.trainDecisionClasses, ['gps_height', 'latitude', 'longitude'])


    def preprocessingFlow1(self):
        self.monthYearOrdinalFromDate()
        #self.monthYearToDummy()
        self.constructionDateReplaceNulls()
        self.publicMeetingNullsToFalse()
        self.permitNullsToFalse()
        self.replaceWrongValuesInLatitudeByNan() 
        self.replaceWrongValuesInLongitudeByNan() 
        self.replaceWrongValuesInPopulationByNan() 
        self.replaceWrongValuesInGpsHeightByNan()
        self.replaceUnknownValuesBySmartMean()  
        self.removeColumns()
        self.replaceTooRareValuesInStringColumns()
        self.replaceUnknownValuesInStringColumns()
        self.encodeAllStringColumns()
        #self.createDummyFromStringColumns()      
    
    def saveResult(self, predictions, fileName='predictions'):
        predictionsColumn = pd.DataFrame(predictions, columns = [self.testDecisionClasses.columns[1]])
        del self.testDecisionClasses['status_group']
        self.testDecisionClasses = pd.concat((self.testDecisionClasses, predictionsColumn), axis = 1)
        DataHandler.saveData({fileName:self.testDecisionClasses})

    def classificationFlow1(self):
        randomForestClassifier = RandomForestClassifier(criterion='gini',
                                min_samples_split=6,
                                n_estimators=1000,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
                            
        randomForestClassifier.fit(self.trainFeatures, self.trainDecisionClasses['status_group'])
        predictions = randomForestClassifier.predict(self.testFeatures) 
        
        print('Szacowana dokładność klasyfikacji {0:.5f}'.format(randomForestClassifier.oob_score_))
        print('Istotność atrybutów\n',pd.DataFrame({'Column':self.trainFeatures.columns,'Importance':randomForestClassifier.feature_importances_}))
                
        return predictions


    def executeExperiment(self, dataPreprocessingFunction, classificationFunction, experimentName):
        print('Eksperyment ' + experimentName)
        self.setData(DataHandler.readData())
        dataPreprocessingFunction()
        DataHandler.saveData({experimentName+'ProcessedTrainingFeatures':self.trainFeatures})
        predictions = classificationFunction()
        self.saveResult(predictions,experimentName+'Predictions')

        
experimentExecutor = ExperimentExecutor()
experimentExecutor.executeExperiment(experimentExecutor.preprocessingFlow1,experimentExecutor.classificationFlow1,'TESTOWY_EKSPERYMENT')



#-------------------------------DEBUGER-------------------------------
# for column in featuresPreprocessor.trainFeatures.columns:
#     #print('Value counts\n',featuresPreprocessor.trainFeatures[column].value_counts())
#     for row in featuresPreprocessor.trainFeatures[column]:
#         #print(column,row)
#         if np.isinf(row) or np.isnan(row):
#             print('ERROR INF OR NAN', column,row)
#         if row == None:
#             print('ERROR NONE', column,row)
#         if isinstance(row, str):
#             print('ERROR STRING', column,row)
# print('--------------KONIEC DEBUGOWANIA-------------------------------')
#-----------------------------------------------------------------------------





    



