"""Machine Learning Pipeline for equity portfolio smart beta momentum factor
    using only price data for robust, parsimonius performance"""

# Import libraries
import math
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
import talib as ta #http://mrjbq7.github.io/ta-lib/func.html
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.grid_search import GridSearchCV

#Import raw data
def get_price_data(fileName):  
    """Read price data from excel into a dataframe.IMPORTANT: Format is dates on leftmost 
    column followed by columns of price data with tickers along top row"""
    
    price_data = pd.read_excel(fileName)
    
    #Set dates as dataframe index and delete 'Date' comun
    price_data.index = price_data['Date']
    price_data.drop('Date', axis=1, inplace=True)
    del price_data.index.name #Formating: remove column name of index.
    
    print "Data set loaded, dimensions: {}".format(price_data.shape)
    return price_data

#Rank and lassify stocks by returns
def label_stock_returns(price_data, retDays, top, rankType):
    """Generate time series of returns for each stock."""
    
    returns = pd.DataFrame(index=price_data.index) #initialise dataframe for returns
    labels = pd.DataFrame(index=price_data.index)  #initialise dataframe for Y labels
    
    for column in price_data: #loop trough each stock
        returns[column] = price_data[column].pct_change(retDays) #pct returns
        
    returns = returns.rank(axis=1,pct=True) #rank reuturns
        
    #Convert rank into a classifier label
    if rankType == 'binary': #Identify only outperforming stocks 'top'
        returns = returns.applymap(lambda x: 'top' if x >= top  
                        else 'NaN' if math.isnan(x) #Catch NaNs
                        else 'bottom')
        
    else: #3-way ranking
        returns = returns.applymap(lambda x: 'top' if x >= top 
                        else 'bottom' if x <= (1.0-top) 
                        else 'NaN' if math.isnan(x) 
                        else 'mid')
    
    #Transform into training labels by adding column labels and shifting in line with features
    
    for column in price_data:
        labels[str(column+'_Ylabel')] = returns[column].shift(-retDays)    
    print "Stock labels complete. Dataframe size: {}".format(labels.shape)
    return labels


#High pass filter - pre processing
def hp_filter(price_data, span):
    """Apply exponentially weighted moving average as noise filter"""
    ewma = pd.stats.moments.ewma #instiate ewma object
    filt = pd.DataFrame(index=price_data.index) #initialise dataframe for filtered prices
    
    for column in price_data:
        filt[column] = ewma( price_data[column], span)
    print "Applied {}day ewma filter to all time series data".format(span)
    
    return filt


#Historic returns
def stock_returns(price_data, lookbackTenors, rankFlag):
    """Generate x-sectional historic returns. Can return rank or value"""
    returns = pd.DataFrame(index=price_data.index) #initialise dataframe for returns
    retRank = pd.DataFrame(index=price_data.index) #initialise dataframe for return ranking
    
    name = 'retRk' if rankFlag == 1 else 'ret' #feature name post-fix
    
    for tenor in lookbackTenors:
        for column in price_data:
            returns[column] = price_data[column].pct_change(tenor)
        
        if rankFlag:    
            returns = returns.rank(axis=1,pct=True) #rank returns

        for column in price_data: #loop back through each stock and name feature
            retRank[str(column+'_'+str(tenor)+name)] = returns[column]
    
    print "Stock returns complete. Dataframe size: {}".format(retRank.shape)
    return retRank

def reversal_techs(price_data, rankFlag):
    """Rversal technicals"""
    revTechs = pd.DataFrame(index=price_data.index)
    rev = pd.DataFrame(index=price_data.index)
    tenors = [5,22,260]
    
    for i in range(len(tenors)-1):
        for column in price_data:#loop through stocks

            longTenor = price_data[column].pct_change(tenors[i+1]) 
            shortTenor = price_data[column].pct_change(tenors[i]) 

            rev[column] = longTenor - shortTenor
            
        if rankFlag:    
            rev = rev.rank(axis=1,pct=True) #rank reversals
    
        for column in price_data: 
            revTechs[str(column+'_'+str(tenors[i+1])+'-'+str(tenors[i])+'rev')] =rev[column]
    
    print "Reversal technicals dataframe size: {}".format(revTechs.shape)
    return revTechs

# Moving average divergence convergence indicators
def macd_Indicators(price_data,lookbackTenors, smooth, adjacentPairs, rankFlag):
    macdTechs = pd.DataFrame(index=price_data.index)
    macdSeries = pd.DataFrame(index=price_data.index)
    """Calc macd for adjacent tenor intervals in lookback tenors
    'smooth' variable is EWMA for signal."""
    
    for i in range(len(lookbackTenors)-1):
        for column in price_data:
            spot = np.array([float(x) for x in price_data[column]]) #TA-lib needs float data in numpy array.
            
            #if adjacentPairs =1 use adjacent tenors for ewma in macd otherwise fastest vs all others
            macd = ta.MACD(spot, lookbackTenors[i*adjacentPairs], lookbackTenors[i+1], signalperiod=smooth)
            macdSeries[column] = macd[2][:] #use signal only from TA-Lib output
        
        if rankFlag:    
            macdSeries = macdSeries.rank(axis=1,pct=True) #rank macd
            
            #name column as stock + tenor + indicator. Use '_' as hierachy break. Calc indicator.
        for column in price_data:
            macdTechs[str(column+'_'+str(lookbackTenors[i+1])+'macd')] = macdSeries[column]
        
            
    print "macd technicals dataframe size: {}".format(macdTechs.shape)
    return macdTechs

# Moving average divergence convergence indicators
def macd_sum(price_data,lookbackTenors, smooth, adjacentPairs, rankFlag):
    macdTechs = pd.DataFrame(index=price_data.index)
    """Calc time scaled macd sum for lookback tenors."""
    
    for column in price_data:
        macdSum = 0.0 #initialise sum feature
        for i in range(len(lookbackTenors)-1):
            spot = np.array([float(x) for x in price_data[column]]) #TA-lib needs float data in numpy array.
            
            #if adjacentPairs =1 use adjacent tenors for ewma in macd otherwise fastest vs all others
            macdSeries = ta.MACD(spot, lookbackTenors[i*adjacentPairs], lookbackTenors[i+1], signalperiod=smooth)
            macdTechs[str(column+'_'+'1macdSum')] = math.sqrt(lookbackTenors[i+1])*macdSeries[2][:] + macdSum #sum all rng indicators / stock
    
    if rankFlag:    
            macdTechs = macdTechs.rank(axis=1,pct=True, numeric_only=True)

    print "macd sum technicals dataframe size: {}".format(macdTechs.shape)
    return macdTechs

#Xsectional ranking momentum 
def xsect_mom(price_data, rankedReturns, lookbackTenors):
    """Change in stock return ranking over time. 
    Needs output from stock_returns with rankFlag =1"""
    
    xsectChg = pd.DataFrame(index=rankedReturns.index) #initialise dataframe for delta rank
    xsectMom = pd.DataFrame(index=rankedReturns.index) #initialise dataframe for output data
    
    for i in range(len(lookbackTenors)-1):
        for column in price_data: 
            
            #need to index 'pure' stock tickers from previously generated ranked returns
            xsectChg[column] = rankedReturns[str(column+'_'+str(lookbackTenors[i+1])+'retRk')].subtract(
                rankedReturns[str(column+'_'+str(lookbackTenors[i])+'retRk')])
        
        xsectChg = xsectChg.rank(axis=1,pct=True) #rank change    
        
        for column in price_data:
            xsectMom[str(column+'_'+str(lookbackTenors[i])+'xsctMom')] = xsectChg[column]
    
    print "X-sectional momentum ranking complete. Dataframe size: {}".format(xsectMom.shape)
    return xsectMom


####################################################################
# Data munging
####################################################################

#Combine feture set and lables 

def combine_features_labels(featureSets,labels):
    """Combine features and labels into one dataFrame."""
    
    featureSets.append(labels)
    combData = pd.concat(featureSets, axis=1)
    
    print "Combined featues and label dataframe size: {}".format(combData.shape)
    return combData

#Trim lead and lag NaNs
#Fix *args
class trim_data:
    """Find start and end dates of useable data and trim. *args for volatility
    tenor if using vola indicators"""
    def __init__(self, price_data, allData, lookbackTenors, retDays, *args): #volaDays
        #volaDays = 0 if args[0] is None else args[0] #Check if we are using vola indicators
        volaDays = 0 #temp fix
        
        self.dataStart = price_data.index[lookbackTenors[-1]+volaDays]
        self.dataEnd = price_data.index[-(retDays+3)]
        self.data = allData.ix[self.dataStart:self.dataEnd]  #trim off indictor run in period
        
        print "Start Date: " + (self.dataStart.strftime("%Y-%m-%d"))
        print "End Date: " + (self.dataEnd.strftime("%Y-%m-%d"))
        
        
#Get train / test split date
def get_split_Date(testFraction, dataSet):
    """Get train / test split as date for a given test fraction"""
    
    splitDate = dataSet.index[-dataSet.shape[0]*testFraction]
    
    print("split date " + splitDate.strftime("%Y-%m-%d"))
    
    return splitDate

#Pivot data (stack) so that all training examples are stacked in standad X matrix
def stack_features(allData):
    """"Impose 2 level hierachy on colums: stock ticker then indicators. Split column 
    names at '_' for new labels. Stack individual stock features by date  """
    
    #Spit colum hiearchy in stocks then sub heading indicators
    allData.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in allData.columns])
    
    #Stack by TOP hierachy (stock name) 
    allData = allData.stack(0) #indicator sets are stacked by date then stock.
    
    print "Stacked data set dimensions: {}".format(allData.shape)
    return allData


#Clean remaining NaNs for stock that cease trading during sample period
def clean_NaNs(allData):
    """Drop rows containing NaNs. This is where stocks have stopped trading"""
    
    allData = allData.dropna()
    allData = allData[allData.Ylabel.str.contains("NaN") == False] #more reliable than dropna
    
    print "Cleaned full data set dimensions: {}".format(allData.shape)
    return allData


#Some basic statistics on feature set
def feature_stats(allData):
    """Basic data exploration - examine before scaling"""
    
    print "Feature means: \n{}".format(np.mean(allData,0)) #all colums except last (label)
    print "Feature standard deviation: \n{}".format(np.std(allData,0))
    print "Feature max: \n{}".format(np.max(allData,0))
    print "Feature min: \n{}".format(np.min(allData,0))

    
#remove mid 
def remove_mid(allData):
    """Drop rows containing mid to provide only salient examples of out / underperformance"""
    
    allData = allData[allData.Ylabel.str.contains('mid') == False] #more reliable than dropna
    
    print "Dropped all mid examples: {}".format(allData.shape)
    return allData

#Feature scaling 
def scale_features(allData, labelsFlag):
    """Scales feature set. Will ignore last column in dataset if labelsFlag is True"""
    
    if labelsFlag:
        allData.iloc[:,:-1] = preprocessing.scale(allData.iloc[:,:-1],copy=False)
    
    else:
        allData = preprocessing.scale(allData,copy=False)
    
    print "Data has been scaled"
    
    return allData

#Feature decomposition (TODO: option to chart variance ratio and auto select n features based on this)
#does labels flag make sense on data - rejoin. Surely if....
def feature_decomposition(allData, decomp_method, components, labelsFlag):
    """Decompose features: decomp_method = 1 for PCA, 2 for ICA, 0 for none"""
    
    labelsFlag = 1 if labelsFlag else 0
    
    decompData = pd.DataFrame()
    
    if decomp_method == 1: #PCA
        decomp = PCA(components,copy=False)
    
    elif decomp_method ==2: #ICA
        decomp = FastICA(components)
    
    else:
        print "No decompostion performed"
        return allData
    
    decomp.fit(allData.iloc[:,:-labelsFlag])
    #print decomp.components_
    #print decomp.explained_variance_ratio_
    
    decompData = pd.DataFrame(decomp.transform(allData.iloc[:,:-labelsFlag]),index = allData.index ) #get pca reduced data. Apply Y index
    
    if labelsFlag ==1:
        decompData = pd.concat([decompData, allData.iloc[:,allData.shape[1]-1]],axis=1)
    
    print "Training set dimensions after decompostion: {}".format(decompData.shape)
    return decompData

# Check labels
def check_labels(allData,labelCol):
    """Check for NaNs in label data"""
    
    number_of_y_NaNs = allData[labelCol].str.contains(r'NaN').sum()
    
    if number_of_y_NaNs:
        print "number ofNaNs in Y labels is: " + str(number_of_y_NaNs)
    
    else:
        print"Labels are NaN free"

#Final split of data into train/test & X/Y
class train_test_split:
    """Returns object with methods that produce various data configurations"""
    def __init__(self, allData, dataStart, dataEnd, splitDate, labelCol): 
        #Create multilevel hiearchy dataframes for train and test sections
        
        #This is quite convoluted, but you cant slice a multilevel dataframe without the 
        #resulting datframe inheriting the WHOLE index of the source multihierachy dataframe
        
        trainData = allData.unstack(-1)
        trainData = trainData.swaplevel(0, 1, axis=1)
        trainData = trainData.ix[dataStart:splitDate]
        trainData = trainData.stack(0)
        
        testData = allData.unstack(-1)
        testData = testData.swaplevel(0, 1, axis=1)
        testData = testData.ix[(splitDate+ pd.DateOffset(1)):dataEnd]
        testData = testData.stack(0)        
        
        #methods
        self.y_train = trainData[labelCol]
        self.X_train = trainData.drop(labelCol, 1)
        self.y_test = testData[labelCol]
        self.X_test = testData.drop(labelCol, 1)
        self.splitDate = splitDate
        self.allTrain = trainData
        self.allTest = testData
        self.allData = allData
        
        print "Data split complete"

        

#################################################
#Machine Learning
#################################################
        
def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    clf.fit(X_train, y_train)

    
def predict_labels(clf, features, target):
    y_pred = clf.predict(features)
    return accuracy_score(target.values, y_pred)


def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    #print "Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    print "Train start {}".format(X_train.index[0][0])
    print "Train end {}".format(X_train.index[-1][0])
    print "Test start {}".format(X_test.index[0][0])
    print "Test end {}".format(X_test.index[-1][0])
    print "Accuracy score for training set: {}".format(predict_labels(clf, X_train, y_train))
    print "Accuracy score for test set: {}".format(predict_labels(clf, X_test, y_test))

    
def run_gridsearch(clf, parameters, X_train, y_train, X_test, y_test):
    """cross-validated optimised parameter search"""
    start = time.time()
    
    #Scorer object
    scorer = make_scorer(accuracy_score, greater_is_better=True)
    
    #Gridsearch
    tuned_clf = GridSearchCV(clf, parameters,scoring=scorer)
    
    print "Final Model: "
    
    tuned_clf.fit(X_train, y_train)
    print "Best Parameters: {:}".format(tuned_clf.best_params_)
    
    #Calculate Accuracy for tuned clasifier
    est = tuned_clf.best_estimator_ 
    tuned_pred = est.predict(X_test)
    
    print "accuracy score for tuned classifier: {:.3f}".format(accuracy_score(y_test, tuned_pred))
    print "Training set: {} samples".format(X_train.shape[0])
    print "Test set: {} samples".format(X_test.shape[0])
    
    end = time.time()
    print "Grid search time (secs): {:.3f}".format(end - start)


#Walk forward cross validation
def walk_forward_days(allData, bd, trainWindow, testWindow, clf, dropTrainMids):
    """Walks train/test window forward in increments"""
    
    #Test we have enough data
    allDays = allData.index.levels[0].shape[0]-trainWindow-testWindow-1
    if allDays < bd: print "Less than one year of walkforward possible!!!"
    
    #initialise variables
    wf = pd.DataFrame({'test_start':[], 'test_end': [], 'score': []})
    startWin = [] #start of training set
    splitWin = [] #end of training set
    splitWinPlus = [] #start of test set (splitWin +1)
    endWin = [] #end of test set
    score = [] #score for each test set 
    
    #create list start,split,end dates for each walkforward step
    for i in range(0,allDays,testWindow):
        endWin.append(allData.index.levels[0][-1-(i)]) #
        startWin.append(allData.index.levels[0][-1-(i+(trainWindow+testWindow))])
        splitWin.append(allData.index.levels[0][-1-(i+testWindow)])
        splitWinPlus.append(allData.index.levels[0][-1-(i+testWindow-1)])
    splitWin.append('Average')
    endWin.append('Average')
    wf.test_end = endWin
    wf.test_start = splitWin
    
    #For each step, construct data sets, train and test
    k = len(startWin)
    
    for i in range(k):
        
        if dropTrainMids ==1:
            focusTrain = remove_mid(allData.ix[startWin[i]:splitWin[i]])
            y_train_Sample = focusTrain['Ylabel']
            X_train_Sample = (focusTrain).drop('Ylabel', 1)
        
        else:
            y_train_Sample = allData.ix[startWin[i]:splitWin[i]]['Ylabel']
            X_train_Sample = (allData.ix[startWin[i]:splitWin[i]]).drop('Ylabel', 1)
        
        y_test_Sample = allData.ix[splitWinPlus[i]:endWin[i]]['Ylabel']
        X_test_Sample = (allData.ix[splitWinPlus[i]:endWin[i]]).drop('Ylabel', 1)
        
        #Train and test
        train_predict(clf, X_train_Sample, y_train_Sample, X_test_Sample, y_test_Sample)
        score.append(predict_labels(clf, X_test_Sample, y_test_Sample))
    
    score.append(np.mean(score))# mean value of score
    wf.score = score
    return wf


def generate_data(dataFile, filterSpan, retDays, lookbackTenors, top, rankType):
    """This function steps through the pipleline. It's only real use is encapsulation when manipulation
    big chunks of data. """
    #Load data
    price_data = get_price_data(dataFile)
    
    #Generate labels. This must be done first on raw data
    labels = label_stock_returns(price_data, retDays, top, rankType)
    
    #Pre-processing: high pass noise filter with ewma
    if filterSpan != 0:
        price_data = hp_filter(price_data, filterSpan)
    
    #Generate Remaining Technical Indicators
    retRank = stock_returns(price_data, lookbackTenors, rankFlag=1)
    ret = stock_returns(price_data, lookbackTenors, rankFlag=0)
    xsectMom = xsect_mom(price_data, retRank, lookbackTenors)
    macdTechs = macd_Indicators(price_data,lookbackTenors, smooth, adjacentPairs =1, rankFlag=1)
    macdSum = macd_sum(price_data,lookbackTenors, smooth, adjacentPairs=1, rankFlag=1)
    revTechs = reversal_techs(price_data, rankFlag=1)
  
    #Choose features
    featureSets = [ret, retRank, revTechs, xsectMom, macdTechs, macdSum]
    allData = combine_features_labels(featureSets,labels)
    ###############
    #Note - structure of all data is dependent on labeling feature types with a number!! (Keeps Ylabel on right)

    #Trim data we know is useless
    trim = trim_data(price_data, allData, lookbackTenors, retDays)   
    dataStart = trim.dataStart
    dataEnd = trim.dataEnd
    allData = trim.data
    
    #Train/test split date
    splitDate = get_split_Date(testFraction, allData)

    #Stack individual sock features into example rows in X matrix
    allData = stack_features(allData)

    #drop rows with NaNs: at this stage only individual stocks affected
    allData = clean_NaNs(allData)
    
    #Feature Scaling
    if scaler == 1:
        allData = scale_features(allData,1)

    #feature deocomposition. Method 1 = PCA, 2 ICA, 0 None
    allData = feature_decomposition(allData, decomp_method, components, labelsFlag)

    #Check integrity of label data
    check_labels(allData,'Ylabel') #double check label data

    #Split data into train / test and X / y
    data = train_test_split(allData, dataStart, dataEnd, splitDate, 'Ylabel', removeMids)
    
    return data

##############################################
# Pipeline Control
##############################################


#Raw data inputs
dataFile = "SPX.xlsx" #Filename - should be in same Dir as this .py 
#Alt Files: SPX_debug.xlsx SPX_debug_full_length.xlsx SPX_light.xlsx SPX_start.xlsx SPX_trunc.xlsx "SPX_stocks.xlsx"

# Noise filter
filterSpan = 1# Span in days. Set to 0 for no filtering.

#Calssification structure
rankType = 'binary'#'binary' #Labeling. Use 'multi' outer percentile ranges (top 25%, middle 50%, bottom 25%)
retDays = int(252*0.75) #'lookforward' tenor for stock performance classifiaction returns
top = 0.50 #Classification boundary (Binary separator or Outer percentile for multi class)
dropTrainMids = 0 # 0 or 1. If 1 remove data classified as mid. Only applicable if rankeType is !=binary

# Input technical analysis feature varaibles
lookbackTenors = [5,23,67,135,252,378]# std

#indSum = 1 #Include time weighted sum of momentum indicators
smooth = 3 #Indicator output signal smoothing where applicable
testFraction = 0.3 #Fraction of dataset to use in test set

#General processing
scaler = 1  #1 to implement, 0 otherwise

#decompostion methods
decomp_method = 0 #0=none, 1=PCA, 2=ICA
components = 8 #Number of features after feature reduction
labelsFlag =1 #For this pipleline always 1, but could set 0 if reusing code


#data has attributes X_train, y_train, X_test, y_test, allTrain, allTest, allData, splitDate    
data = generate_data(dataFile, filterSpan, retDays, lookbackTenors, top, 
                        rankType)

##########################################
#Walk Forward Cross Validation
##########################################

from sklearn.ensemble import RandomForestClassifier
bd = 252# number of business days in year
trainWindow = int(3*bd) #days
testWindow = int(bd/12) #days

clf = RandomForestClassifier(n_estimators=100,  max_features='auto',  max_depth=2, min_samples_split=500, n_jobs=-1)

#wf = walk_forward_days(data.allTrain , bd, trainWindow, testWindow, clf, dropTrainMids)
wf = walk_forward_days(data.allData , bd, trainWindow, testWindow, clf, dropTrainMids)
print wf
