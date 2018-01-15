import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
import pylab
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class FeaturesPreprocessorHelper:
    # -------------Helpers Functions--------------------

    # Funkcja pomocnicza tworząca nową kolumne 'roku' z kolumny daty
    @staticmethod
    def yearFromDate(trainFeatures, testFeatures, dateColumn, yearColumn):
        for dataset in [trainFeatures, testFeatures]:
            dataset[dateColumn] = pd.to_datetime(dataset[dateColumn])
            
            yearRecorded = []
            for date in dataset[dateColumn]:
                yearRecorded.append(date.year)
            dataset[yearColumn] = pd.Series(yearRecorded)         
        return trainFeatures, testFeatures

    # Funkcja pomocnicza tworząca nową kolumne 'miesiąca' z kolumny daty
    @staticmethod
    def monthFromDate(trainFeatures, testFeatures, dateColumn, monthColumn):
        for dataset in [trainFeatures, testFeatures]:
            dataset[dateColumn] = pd.to_datetime(dataset[dateColumn])
            
            monthRecorded = []
            for date in dataset[dateColumn]:
                monthRecorded.append(date.month)
            dataset[monthColumn] = pd.Series(monthRecorded)         
        return trainFeatures, testFeatures

    # Funkcja zmieniająca datę na porządkową
    @staticmethod
    def ordinalFromDate(trainFeatures, testFeatures, dateColumn, ordinalDateColumn):
        for dataset in [trainFeatures, testFeatures]:
            dataset[dateColumn] = pd.to_datetime(dataset[dateColumn])
            dataset[ordinalDateColumn] = dataset[dateColumn].apply(lambda x: x.toordinal())       
        return trainFeatures, testFeatures

    # Funkcja pomocnicza rozbijająca date na dodatkowe kolumny- rok, miesiąc, datę porządkową
    @staticmethod
    def monthYearOrdinalFromDate(trainFeatures, testFeatures, dateColumn, yearColumn, monthColumn, ordinalDateColumn):
        for dataset in [trainFeatures, testFeatures]:
            #print(dataset)
            dataset['date_recorded'] = pd.to_datetime(dataset['date_recorded'])
            
            year_recorded = []
            month_recorded = []
            for date in dataset[dateColumn]:
                year_recorded.append(date.year)
                month_recorded.append(date.month)
            dataset[yearColumn] = pd.Series(year_recorded)         
            dataset[monthColumn] = pd.Series(month_recorded)

            dataset[ordinalDateColumn] = dataset[dateColumn].apply(lambda x: x.toordinal())
        return trainFeatures, testFeatures

    # Funkcja pomocnicza do tworzenia kolumn fikcyjnych, na podstawie kolumn podanych jako parametr
    # Nowe kolumny tworzone są na podstawie wartości, które wstępują (we wskazanej kolumnie) zarówno w zbiorze testowym jak i treningowym
    @staticmethod
    def toDummy(trainFeatures,testFeatures,columns):
        for column in columns:
            #trainFeatures[column] = trainFeatures[column].apply(lambda x:str(x)) 
            #testFeatures[column] = testFeatures[column].apply(lambda x:str(x))

            uniqueTrainValues = trainFeatures[column].unique()
            uniqueTestValues = testFeatures[column].unique()
            commonUniqueValues = []
            for value in uniqueTrainValues:
                if value in uniqueTestValues:
                    commonUniqueValues.append(column+'_'+str(value))

            dummyForTrainFromCommonValues = pd.get_dummies(trainFeatures[column], prefix=column)[commonUniqueValues]        
            trainFeatures=pd.concat((trainFeatures,dummyForTrainFromCommonValues),axis=1)

            dummyForTestFromCommonValues = pd.get_dummies(testFeatures[column], prefix=column)[commonUniqueValues]
            testFeatures=pd.concat((testFeatures,dummyForTestFromCommonValues),axis=1)
        return trainFeatures, testFeatures


    # Funkcja pomocnicza do zastępowania wskazanej wartości w kolumnach, wartością podaną
    @staticmethod
    def replaceValueByValue(trainFeatures, testFeatures, columns, toBeReplaced, replacing):
        for dataset in [trainFeatures, testFeatures]:
            for column in columns:
                dataset[column].replace(toBeReplaced, replacing, inplace=True)
        return trainFeatures, testFeatures


    # Funkcja pomocnicza do zastępowania wskazanej wartości w kolumnie, wartościami średnimi
    @staticmethod
    def replaceValueByMean(trainFeatures, testFeatures, columns, toBeReplaced):
        for dataset in [trainFeatures, testFeatures]:
            for column in columns:
                meanValue = round(dataset[dataset[column]!=toBeReplaced][column].mean())
                dataset[column].replace(toBeReplaced, meanValue, inplace=True)
        return trainFeatures, testFeatures

    # Funkcja pomocnicza do zastępowania nieznanych wartości w kolumnie, wartościami średnimi
    # W założeniu kolumna zawiera wartości numeryczne, wartości nieznane oznacz
    @staticmethod
    def replaceUnknownByMean(trainFeatures, testFeatures, columns):
        for dataset in [trainFeatures, testFeatures]:
            for column in columns:
                meanValue = round(dataset[column].mean())
                toBeReplaced=[None, np.nan]
                dataset[column].replace(toBeReplaced, meanValue, inplace=True)
        return trainFeatures, testFeatures

    @staticmethod
    def replaceWrongByMean(trainFeatures, testFeatures, columns):
        for dataset in [trainFeatures, testFeatures]:
            for column in columns:
                meanValue = round(dataset[column].mean())
                toBeReplaced=[None, np.nan]
                dataset[column].replace(toBeReplaced, meanValue, inplace=True)
        return trainFeatures, testFeatures

    # Funkcja pomocnicza aplikująca daną funkcję na wszystkich wartościach wskazanych kolumn
    @staticmethod
    def applyFunctionOnColumns(trainFeatures, testFeatures, columns, function):
        for dataset in [trainFeatures, testFeatures]:
            for column in columns:
                dataset[column] = dataset[column].apply(function)
        return trainFeatures, testFeatures

    # Funkcja wstawiająca w puste miejsca w kolumnie, jej wartością średnią zależną od wartości innych kolumn
    # Średnia dla danego wiersza jest wybierana na podstawie parametrów kolumn 'zależnych'
    # @staticmethod
    # def replaceColumnIndicatedValuesByMeanDependedOnOtherColumns(trainFeatures, testFeatures, columns, baseColumns):
    #     for column in columns:
    #         print('OriginalTrain\n',trainFeatures[column])
    #         #print('OriginalTest\n',trainFeatures[column])
    #         meanColumnForTrain = pd.Series(np.zeros(len(trainFeatures[column])))
    #         print('meanColumnForTrain\n',meanColumnForTrain)
    #         meanColumnForTest = pd.Series(np.zeros(len(testFeatures[column])))
    #         for baseColumn in baseColumns:
    #             trainGroupedMean = trainFeatures.groupby(baseColumn)[column].mean().rename(str(column)+'mean').reset_index()
    #             referencedMean = pd.merge(trainFeatures, trainGroupedMean).iloc[:,-1]          
    #             referencedMean = pd.Series(referencedMean)
    #             referencedMean.replace(np.nan, referencedMean.mean(),inplace=True) #Tutaj jest taki manewr, że po grupowaniu i tak średnia dla braukjących wartości może wynosić nan, bo wszystkie wartości przy liczeniu średniej dla danej grupy wynosiły nan, trzeba to uzupełnić średnią   
    #             print('Referenced\n',referencedMean)
    #             meanColumnForTrain = meanColumnForTrain.add(referencedMean)
    #             print('meanColumnForTrain\n',meanColumnForTrain)

    #             trainGroupedMean = trainFeatures.groupby([baseColumn])[column].mean()
    #             trainDFGroupedMean = pd.DataFrame(trainGroupedMean)
    #             referencedMean = pd.merge(testFeatures, trainDFGroupedMean, left_on=[baseColumn], right_index=True,how='left').iloc[:,-1]
    #             referencedMean.replace(np.nan, referencedMean.mean(),inplace=True) #Tutaj jest taki manewr, że po grupowaniu i tak średnia dla braukjących wartości może wynosić nan, bo wszystkie wartości przy liczeniu średniej dla danej grupy wynosiły nan, trzeba to uzupełnić średnią
    #             meanColumnForTest = meanColumnForTest.add(referencedMean)

    #         meanColumnForTrain = meanColumnForTrain.apply(lambda x : x/len(baseColumns))
    #         print('meanColumnForTrain\n',meanColumnForTrain)
    #         meanColumnForTest = meanColumnForTest.apply(lambda x : x/len(baseColumns))
            
    #         trainFeatures[column] = trainFeatures[column].fillna(meanColumnForTrain)
    #         print(str(column)+'TRAIN\n',trainFeatures[column])
    #         testFeatures[column] = testFeatures[column].fillna(meanColumnForTest)

    #         # for row in meanColumnForTrain:
    #         #     #print(column,row)
    #         #     if np.isinf(row) or np.isnan(row):
    #         #         print('ERROR', column,row)
    #         # print('Koniec feature processingu @@@@@@@@@@@@@')
    #         #print(str(column)+'TEST\n',testFeatures[column])
    #         #print(str(column)+'TRAIN\n',trainFeatures[column])
    #     return trainFeatures, testFeatures

    @staticmethod
    def replaceColumnIndicatedValuesByMeanDependedOnOtherColumns(trainFeatures, testFeatures, columns, baseColumns):
        for column in columns:
            #print('OriginalTrain\n',trainFeatures[column])
            #print('OriginalTest\n',trainFeatures[column])
            meanColumnForTrain = pd.Series(np.zeros(len(trainFeatures[column])))
            meanColumnForTest = pd.Series(np.zeros(len(testFeatures[column])))
            for baseColumn in baseColumns:
                trainGroupedMean = trainFeatures.groupby(baseColumn)[column].mean().rename(str(column)+'mean').reset_index()
                referencedMean = pd.merge(trainFeatures, trainGroupedMean, how='left').iloc[:,-1]          
                referencedMean = pd.Series(referencedMean)
                referencedMean.replace(np.nan, referencedMean.mean(),inplace=True) #Tutaj jest taki manewr, że po grupowaniu i tak średnia dla braukjących wartości może wynosić nan, bo wszystkie wartości przy liczeniu średniej dla danej grupy wynosiły nan, trzeba to uzupełnić średnią   
                meanColumnForTrain = meanColumnForTrain.add(referencedMean)
        
                trainGroupedMean = trainFeatures.groupby([baseColumn])[column].mean()
                trainDFGroupedMean = pd.DataFrame(trainGroupedMean)
                referencedMean = pd.merge(testFeatures, trainDFGroupedMean, left_on=[baseColumn], right_index=True,how='left').iloc[:,-1]
                referencedMean.replace(np.nan, referencedMean.mean(),inplace=True) #Tutaj jest taki manewr, że po grupowaniu i tak średnia dla braukjących wartości może wynosić nan, bo wszystkie wartości przy liczeniu średniej dla danej grupy wynosiły nan, trzeba to uzupełnić średnią
                meanColumnForTest = meanColumnForTest.add(referencedMean)

            meanColumnForTrain = meanColumnForTrain.apply(lambda x : x/len(baseColumns))
            meanColumnForTest = meanColumnForTest.apply(lambda x : x/len(baseColumns))
            
            trainFeatures[column] = trainFeatures[column].fillna(meanColumnForTrain)
            testFeatures[column] = testFeatures[column].fillna(meanColumnForTest)

            #print(str(column)+'TEST\n',testFeatures[column])
            #print(str(column)+'TRAIN\n',trainFeatures[column])
        return trainFeatures, testFeatures
    
    # Usuwa wskazane kolumny
    @staticmethod
    def removeColumns(trainFeatures, testFeatures, columns):
        for column in columns:
            del trainFeatures[column]
            del testFeatures[column]
        return trainFeatures, testFeatures

    # Funkcja pomocnicza - redukcja możliwych wartości danej kolumny, gdy ta wystepuje zbyt żadko- zamiana tej wartości na inną
    @staticmethod
    def reduceFactorLevels(trainFeatures, testFeatures, columns, minOccurances=100, replacingValue='ohter'):
        datasetWithCounts = trainFeatures[columns].apply(lambda x: x.map(x.value_counts()))
        trainFeatures[columns] = trainFeatures[columns].where(datasetWithCounts > minOccurances, replacingValue)
        for column in columns:
            for uniqueValue in testFeatures[column].unique():
                if uniqueValue not in trainFeatures[column].unique():
                    testFeatures[column].replace(uniqueValue, 'other', inplace=True)
        return trainFeatures, testFeatures

    # Funkcja pomocnicza zastępująca wartości nieznane w kolumnach wartością podaną
    @staticmethod
    def replaceUnknownColumnsValues(trainFeatures, testFeatures, columns, raplacingValue = 'other'):
        for dataset in trainFeatures, testFeatures:
            for column in columns:
                dataset[column].fillna(raplacingValue, inplace = True)
                dataset[column].fillna(raplacingValue, inplace = True)
        return trainFeatures, testFeatures


    # Funkcja zastępująca wszystkie wartości nieznane w kolumnach typu string
    @staticmethod
    def replaceUnknownValuesInStringColumns(trainFeatures, testFeatures, raplacingValue = 'other'):
        columns = [i for i in trainFeatures.columns if type(trainFeatures[i].iloc[0]) == str]  
        return FeaturesPreprocessorHelper.replaceUnknownColumnsValues(trainFeatures, testFeatures, columns, raplacingValue)

    # Utworzenie dummy variable (zmiennych fikcyjnych) z wszystkich kolumn typu string
    @staticmethod
    def createDummyFromStringColumns(trainFeatures, testFeatures):
        columns = [i for i in trainFeatures.columns if type(trainFeatures[i].iloc[0]) == str]    
        return toDummy(trainFeatures,testFeatures,columns)


    #jest jeszcze categorical encoder

    # Transformacja wskazanych kolumn categorical na numerical (zawierających stringi, napisy  na wartości numeryczne)
    @staticmethod
    def encodeLabelColumns(trainFeatures, testFeatures, columns):
        labelEncoder = preprocessing.LabelEncoder()
        for column in columns:        
            trainFeatures[column] = labelEncoder.fit_transform(trainFeatures[column])
            testFeatures[column] = labelEncoder.transform(testFeatures[column])
        return trainFeatures,testFeatures

    @staticmethod
    def encodeAllStringColumns(trainFeatures, testFeatures):
        columns = [i for i in trainFeatures.columns if type(trainFeatures[i].iloc[0]) == str]
        return FeaturesPreprocessorHelper.encodeLabelColumns(trainFeatures,testFeatures,columns)


    # Standaryzacja zmiennych w podanych kolumnach
    # Wartości zmiennych są podmieniane, a nazwa kolumny zachowana
    @staticmethod
    def standardScalerStandaryzation(trainFeatures, testFeatures, columns):
        sc = preprocessing.StandardScaler()
        trainStandarized = sc.fit_transform(trainFeatures[columns])
        testStandarized = sc.transform(testFeatures[columns])
        i=0
        for column in columns:
            trainFeatures[column] = trainStandarized[:,i]
            testFeatures[column] = testStandarized[:,i]
            i = i + 1
        return trainFeatures, testFeatures

    #Redukcja wielowymiarowości danych
    @staticmethod  
    def dimensionalityReductionByLinearDiscriminantAnalysis(trainFeatures, testFeatures, trainDecisionClasses, columns):
        lda = LDA(n_components=None)
        trainLDA = lda.fit_transform(trainFeatures[columns], trainDecisionClasses['status_group'])
        testLDA = lda.transform(testFeatures[columns])
        i=0
        for column in columns:
            trainFeatures[column] = trainLDA[:,i]
            testFeatures[column] = testLDA[:,i]
            i = i + 1
        return trainFeatures, testFeatures 

    #Standarzacja wartości z kolumny do podanego zakresu - domyślnie (0,1)
    @staticmethod  
    def dimensionalityReductionByMinMaxScaler(trainFeatures, testFeatures, columns):        
        minMaxScaler = preprocessing.MinMaxScaler()
        trainScaled = minMaxScaler.fit_transform(trainFeatures[columns], trainDecisionClasses['status_group'])
        testScaled = minMaxScaler.transform(testFeatures[columns])
        i=0
        for column in columns:
            trainFeatures[column] = trainScaled[:,i]
            testFeatures[column] = testScaled[:,i]
            i = i + 1
        return trainFeatures, testFeatures 


    @staticmethod 
    def printColumnTypes(datasets):
        for dataset in datasets:
            print('Dataset')
            for column in dataset.columns:
                print('Column:',repr(column).ljust(25),'\ttype:', repr(type(dataset[column].iloc[0])).ljust(25),'\tvalue',dataset[column].iloc[0])

