"""Machine Learning Pipeline for technical analysis stock picking system"""

# Import libraries
import math
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
import talib as ta
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import make_scorer,f1_score #,accuracy_score
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


#Classify stocks by returns

def label_stock_returns(price_data, retDays, top, ordinal, rankType, offset):
    """Generate time series of returns for each stock. Option for ordinal ranking, 
    binary or quartile(default). 'offset' is which the price data starts with first = 0"""
    returns = pd.DataFrame(index=price_data.index) #initialise dataframe for returns
    
    for column in price_data.iloc[:, offset:]: #loop trough each stock
        returns[column] = price_data[column].pct_change(retDays) #pct returns
    #Ordinal ranking
    if ordinal == True:  #Rank returns
        returns = returns.rank(axis=1,pct=True)
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
    labels = pd.DataFrame(index=price_data.index)
    for column in price_data.iloc[:, 1:]:
        labels[str(column+'_Ylabel')] = returns[column].shift(-retDays)    
    print "Stock returns & labels complete. Dataframe size: {}".format(labels.shape)
    return labels


##################################################################################
#Generate features - here we use technical analysis indicators and similar metrics
##################################################################################

def range_technicals(price_data, lookbackTenors, timeSum, offset):
    """Generates a range lookback with current position in range 0,1.
    Option to add timeWeighted sum indivdual indicators"""
    rangeTechs = pd.DataFrame(index=price_data.index)

    #loop trough each stock and lookbackTime. NB miss out first column with index in. 
    for column in price_data.iloc[:, offset:]:
        rngSum = 0.0 #initialise
        
        for tenor in lookbackTenors: 
            spot = price_data[column]
            rolling_min = pd.rolling_min(price_data[column],window=tenor)
            rolling_max = pd.rolling_max(price_data[column],window=tenor)
            rng = (spot-rolling_min)/(rolling_max-rolling_min) #rng indicator
            #name column as stock + tenor + indicator. 
            rangeTechs[str(column+'_'+str(tenor)+'rng')] = rng #Use '_' for potential column hierachy break.
            rngSum = math.sqrt(tenor)*rng + rngSum #sum all rng indicators / stock
        
        if timeSum: #Add final feature: time weighted sum
            rangeTechs[str(column+'_'+'1rngSum')] = rngSum 
            
    print "Range technicals dataframe size: {}".format(labels.shape)
    return rangeTechs


def vola_Techs(price_data, volTenor, lookbackTenors, offset):
    """Generate rolling vol range windows based on 100 day volatility"""
    volaTechs = pd.DataFrame(index=price_data.index)

    for column in price_data.iloc[:, offset:]:
        for tenor in lookbackTenors:
        #use pandas vectorised functions
            returns = price_data[column].pct_change(1) #daily srock price returns
            vola = pd.rolling_std(returns.values, volTenor) #no need to sqrt(t) normalise as only one vol tenor
            rolling_min = pd.rolling_min(vola,window=tenor)
            rolling_max = pd.rolling_max(vola,window=tenor)
            volaRng = (vola-rolling_min)/(rolling_max-rolling_min)
            volaTechs[str(column+'_'+str(tenor)+'vola')] = volaRng
    print "Volatility technicals dataframe size: {}".format(volaTechs.shape)
    return volaTechs


def macd_Indicators(price_data,lookbackTenors, smooth, timeSum, offset):
    macdTechs = pd.DataFrame(index=price_data.index)
    """Calc macd for adjacent tenor intervals in lookback tenors
    'smooth' variable is EWMA for signal. Offset is for 1st data column"""
    #loop through stocks and lookback Times. NB miss out first column with index in. 
    for column in price_data.iloc[:, offset:]:
        macdSum = 0.0 #initialise
        for i in range(len(lookbackTenors)-1):
            spot = np.array([float(x) for x in price_data[column]]) #TA-lib needs float data in numpy array.
            macdSeries = ta.MACD(spot, lookbackTenors[i], lookbackTenors[i+1], signalperiod=smooth)
            #name column as stock + tenor + indicator. Use '_' potential hierachy break. Calc indicator
            macdTechs[str(column+'_'+str(lookbackTenors[i+1])+'macd')] = macdSeries[2][:] #Take signal = FastEWMA - SlowEWMA
            macdSum = math.sqrt(lookbackTenors[i+1])*macdSeries[2][:] + macdSum #sum all rng indicators / stock
        
        if timeSum: #Add final feature: time weighted sum
            macdTechs[str(column+'_'+'1macdSum')] = macdSum
            
    print "macd technicals dataframe size: {}".format(macdTechs.shape)
    return macdTechs


def rsi_indicators(price_data,lookbackTenors, smooth, offset):
    rsiTechs = pd.DataFrame(index=price_data.index)
    """Calc ris for each tenor  in lookback tenors
    'smooth' variable is EWMA for signal. Offset is for 1st data column"""
    #loop through stocks and lookback Times. NB miss out first column with index in. 
    for column in price_data.iloc[:, 1:]:
        rsiSum = 0.0 #initialise
        for tenor in lookbackTenors:
            #TA-lib needs float data in numpy array. Price_data seems to contain a mixture of float & real
            spot = np.array([float(x) for x in price_data[column]])
            #name column as stock + tenor + indicator. Use '_' potential hierachy break. Calc indicator
            rsi = ta.RSI(spot, tenor)/100.0 
            rsiTechs[str(column+'_'+str(tenor)+'rsi')] = rsi  
    print "rsi technicals dataframe size: {}".format(rsiTechs.shape)
    return rsiTechs

####################################################################
# Data munging
####################################################################

#Combine feture set and lables 

def combine_features_labels(featureSets,labels):
    """Combine features and labels into one dataFrame"""
    featureSets.append(labels)
    combData = pd.concat(featureSets, axis=1)
    print "Combined featues and label dataframe size: {}".format(combData.shape)
    return combData

#Trim lead and lag NaNs

class trim_data:
    """Find start and end dates of useable data and trim. *args for volatility
    tenor if using vola indicators"""
    def __init__(self, price_data, allData, lookbackTenors, retDays, *args): #volaDays
        volaDays = 0 if args[0] is None else args[0] #Check if we are using vola indicators
        self.dataStart = price_data.index[lookbackTenors[-1]+volaDays]
        self.dataEnd = price_data.index[-(retDays+3)]
        print "Start Date: " + (self.dataStart.strftime("%Y-%m-%d"))
        print "End Date: " + (self.dataEnd.strftime("%Y-%m-%d"))
        self.data = allData.ix[self.dataStart:self.dataEnd]  #trim off indictor run in period

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

#Some simple statistics on feature set

def feature_stats(allData):
    """Basic data exploration - examine before scaling"""
    print "Feature means: \n{}".format(np.mean(allData.iloc[:,:-1],0)) #all colums except last (label)
    print "Feature standard deviation: \n{}".format(np.std(allData.iloc[:,:-1],0))
    print "Feature max: \n{}".format(np.max(allData.iloc[:,:-1],0))
    print "Feature min: \n{}".format(np.min(allData.iloc[:,:-1],0))

#Feature scaling (TODO: option to call function but not scale)

def scale_features(allData, labelsFlag):
    """Scales feature set. Will ignore last column in dataset if labelsFlag is True"""
    if labelsFlag:
        allData.iloc[:,:-1] = preprocessing.scale(allData.iloc[:,:-1],copy=False)
    else:
        allData = preprocessing.scale(allData,copy=False)
    return allData

#Feature decomposition (TODO: option to chart variance ratio and auto select n features based on this)

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
        return
    decomp.fit(allData.iloc[:,:-labelsFlag])
    #print decomp.components_
    #print decomp.explained_variance_ratio_
    decompData = pd.DataFrame(decomp.transform(allData.iloc[:,:-labelsFlag]),index = allData.index ) #get pca reduced data. Apply Y index
    decompData = pd.concat([decompData, allData.iloc[:,allData.shape[1]-labelsFlag]],axis=1)
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
    """Split complete (X & Y) dataset into train,test feature and label sets"""
    def __init__(self, allData, dataStart, dataEnd, splitDate, labelCol): 
        trainData = allData.ix[dataStart:splitDate] 
        testData = allData.ix[(splitDate+ pd.DateOffset(1)):dataEnd]
        print "Training set dimensions: {}".format(trainData.shape)
        print "Testing set dimensions: {}".format(testData.shape)
        self.y_train = trainData[labelCol]
        self.X_train = trainData.drop(labelCol, 1)
        self.y_test = testData[labelCol]
        self.X_test = testData.drop(labelCol, 1)   

#################################################
#Machine Learning
#################################################
        
def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

    
def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    compare1 = pd.DataFrame(target.values)
    compare2 = pd.DataFrame(y_pred)
    n_pred_top = compare2[0].str.contains(r'top').sum()
    n_pred_bot = compare2[0].str.contains(r'bottom').sum()
    compare1 = pd.concat([compare1, compare2], axis=1)
    print "top predictions:" + str(100*n_pred_top/y_pred.shape[0]) + "%"
    print "bottom predictions:" + str(100*n_pred_bot/y_pred.shape[0])+ "%"
    return f1_score(target.values, y_pred, pos_label='top', average='binary')


def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    #print "Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    print "Train start {}".format(X_train.index[0][0])
    print "Train end {}".format(X_train.index[-1][0])
    print "Test start {}".format(X_test.index[0][0])
    print "Test end {}".format(X_test.index[-1][0])
    print "F1 score for training set: {}".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))

    
def run_gridsearch(clf, parameters, X_train, y_train, X_test, y_test):
    """cross-validated optimised parameter search"""
    #Scorer object
    f1_scorer = make_scorer(f1_score, pos_label='top', greater_is_better=True, average='binary')
    #Gridsearch
    tuned_clf = GridSearchCV(clf, parameters,scoring=f1_scorer)
    print "Final Model: "
    tuned_clf.fit(X_train, y_train)
    print "Best Parameters: {:}".format(tuned_clf.best_params_)
    #Calculate F1 for tuned clasifier
    est = tuned_clf.best_estimator_ 
    tuned_pred = est.predict(X_test)
    print "F1 score for tuned classifier: {:.3f}".format(f1_score(y_test, tuned_pred, pos_label='top',average='binary'))
    print "Training set: {} samples".format(X_train.shape[0])
    print "Test set: {} samples".format(X_test.shape[0])

    
def walk_forward(allData, bd, trainWindow, testWindow, clf):
    """Walks train/test window forward in increments"""
    dates =  allData.index.levels[0]
    years = dates.shape[0]//bd-trainWindow-testWindow-1
    if years < 2: print "Less than one year of walkforward possible!!!"
    #print range(years)

    startWin = [] #start of training set
    splitWin = [] #end of training set
    splitWinPlus = [] #start of test set (splitWin +1)
    endWin = [] #end of test set
    #create list start,split,end dates for each walkforward step
    for i in range(years):
        endWin.append(dates[-1-(i*bd)]) #
        startWin.append(dates[-1-(i*bd+(trainWindow+testWindow)*bd)])
        splitWin.append(dates[-1-(i*bd+testWindow*bd)])
        splitWinPlus.append(dates[-1-(i*bd+testWindow*bd-1)])
    #For each step, construct data sets, train and test
    for i in range(years):
        y_train_Sample = allData.ix[startWin[i]:splitWin[i]]['Ylabel']
        X_train_Sample = (allData.ix[startWin[i]:splitWin[i]]).drop('Ylabel', 1)
        y_test_Sample = allData.ix[splitWinPlus[i]:endWin[i]]['Ylabel']
        X_test_Sample = (allData.ix[splitWinPlus[i]:endWin[i]]).drop('Ylabel', 1)
        train_predict(clf, X_train_Sample, y_train_Sample, X_test_Sample, y_test_Sample)
    return 

##############################################
# Pipeline Control
##############################################

# Input technical analysis feature vairable
retDays = 22
lookbackTenors = [5,12,23,67,135,265]
volaDays = 100

#Get raw price data
price_data = get_price_data("SPX_debug_full_length.xlsx") #Alt: SPX_debug.xlsx SPX_debug_full_length.xlsx SPX_light.xlsx

#Generate features and labels
#Offset should generally be generally set to 0, but my dataset has SPX index in first row which should not be used
labels = label_stock_returns(price_data, retDays, 0.5, True,'binary',1)
rangeTechs = range_technicals(price_data, lookbackTenors, True, 1)
volaTechs = vola_Techs(price_data, volaDays, lookbackTenors, 1)
macdTechs = macd_Indicators(price_data,lookbackTenors, 9, True, 1)
rsiTechs = rsi_indicators(price_data,lookbackTenors, smooth = 3, offset = 1)

#Choose features
featureSets = [macdTechs,rangeTechs,rsiTechs,volaTechs]

#Combine feature sets
allData = combine_features_labels(featureSets,labels)

##Todo - double check NaN dropping function. Does it drop on lowest hierachy?
trim = trim_data(price_data, allData, lookbackTenors, retDays, volaDays)   
dataStart = trim.dataStart
dataEnd = trim.dataEnd
allData = trim.data

#Train/test split date
splitDate = get_split_Date(testFraction = 0.3, dataSet = allData)

#Stack individual sock features into example rows in X matrix
allData = stack_features(allData)

#drop rows with NaNs: at this stage only individual stocks affected
allData = clean_NaNs(allData)

#Feature Scaling
allData = scale_features(allData,1)

#feature deocomposition. Method 1 = PCA, 2 ICA, 0 None
allData = feature_decomposition(allData, decomp_method = 1, components = 5, labelsFlag =1)

#Check integrity of label data
check_labels(allData,'Ylabel') #double check label data

#Split data into train and test samples
tt = train_test_split(allData, dataStart, dataEnd, splitDate, 'Ylabel')
y_train = tt.y_train
X_train = tt.X_train
y_test = tt.y_test
X_test = tt.X_test

##########################################
#single train and test on entire data set
##########################################
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=7, min_samples_split=100) #max_depth=7, min_samples_split=100
train_classifier(clf, X_train, y_train)

#Return accuracy score on training set
train_f1_score = predict_labels(clf, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score)
#Predict on test data
print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))

##########################################
#Cross validated grid search
##########################################
clf = tree.DecisionTreeClassifier()
parameters = {'max_depth':(7,8),'min_samples_split':(100,200)}
run_gridsearch(clf, parameters, X_train, y_train, X_test, y_test)

##########################################
#Yearly walk forward
##########################################
bd = 253# number of business days in year
trainWindow = 3 #'3Y'
testWindow = 1 #'1Y'
clf = tree.DecisionTreeClassifier(max_depth=7, min_samples_split=100)
walk_forward(allData, bd, trainWindow, testWindow, clf)

