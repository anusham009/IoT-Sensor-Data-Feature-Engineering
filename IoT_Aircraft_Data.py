# Importing the necessary packages 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.ensemble import ExtraTreesClassifier

# Read the train, test and truth data on to dataframe
url_train = 'http://azuremlsamples.azureml.net/templatedata/PM_train.txt'
df_train = pd.read_csv(url_train, header=None, sep='\s+')
url_test = 'http://azuremlsamples.azureml.net/templatedata/PM_test.txt'
df_test = pd.read_csv(url_test, header=None, sep='\s+')

#Truth data is the remaining working cycles for the test data
url_truth = 'http://azuremlsamples.azureml.net/templatedata/PM_truth.txt'
df_truth = pd.read_csv(url_truth, header=None, sep='\s+')

# The data as such do not have columns 
# Note the column names are added separately and the domain expertise for arriving at the target variable 
#are adapted from sample kaggle data of same kind
col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
df_train.columns=col_names
df_train.head()
df_test.columns=col_names
df_test.head()

#************* Deriving the Target variable in train and test data ***********
# This is also part of feature engineering and requires domain knowledge********

# Here a new column called ttf (time to failure) is derived on the basis of id and cycle 
# Max value of the cycle is the considered as the failure point of engine
# Eg : forid =1 ie. first engine the failure point is 192 ie. next value of max(cycle)
df_train['ttf'] = df_train.groupby(['id'])['cycle'].transform(max)-df_train['cycle']
df_train.head()

# Providing a name for truth column and adding id column as well since 
# this data has to be joined with test data based on id
col_name = ['more']
df_truth.columns = col_name
df_truth['id']=df_truth.index+1

# Generate a new temp data frame 

# generate  a new data frame for column max for test data
new = pd.DataFrame(df_test.groupby('id')['cycle'].max()).reset_index()
new.columns = ['id', 'max']
new.head()

# Generate rtf from more parameter of truth and max parameter of the new table 
df_truth['rtf']=df_truth['more']+new['max']

# Dropping the rul value from truth data frame as we have calculated rtf which is to merged with test
df_truth.drop('more',axis=1, inplace = True)

# Merge the truth data frame to test data frame
df_test=df_test.merge(df_truth,on=['id'],how='left')
df_test['ttf']=df_test['rtf'] - df_test['cycle']
df_test.drop('rtf', axis=1, inplace=True)

# Derviving the label variable to predict whether an engine needs maintanence with in next 30 days or not
period = 30
df_train['label_pred'] = df_train['ttf'].apply(lambda x: 1 if x <= period else 0)
df_train.head()

# Now the data has 27 dependent variables and 1 target variable 'ttf'
# The problem in hand is to predict whether an engine will fail or not in the next n days (n=30 here)

# Check for null values
# No columns have null values in train and test
df_train.isnull().any()
df_test.isnull().any()

# visualize data # columns 13 and 18 have some extreme values but since all values are extreme in the column this cannot be treated as outliers, Rest of the columns have no big variations.
# Hence no outlier treatement is done on the data as the outlier values are not consulted with domain expert.
df_train.boxplot()

# ************* Feature Enginering  ******************

# There is no more data clearning steps performed here as these are sensor readings with no null values and genuine outliers
# Feature engineering is performed using multiple ways
# Correlation ,Recursive Feature Elimination, PCA,Select K best ,Borutapy and Random forest classifier are the methods tried here
# Correlation of all variables 

#************************* Correlation with target variable ***************8
corr = df_train.corr()

# Generate the correlation heatmap 
sns.heatmap(corr)

# Retain features which are ihighly correlated with the target variable 
cor_target = abs(corr["label_pred"])
#Selecting highly correlated features with ttf 
# This ends with 13 variables being significant out of the 26 variables
relevant_features = cor_target[cor_target>0.5]
relevant_features

##Selected Variables :Cycle,s2,s3,s4,s7,s8,s11,s12,s13,s15,s17,s20,s21,ttf

# ******************Using Recursive Feature Elimination**************************8
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# Select top n features : Have just chosen 15 features here
# Feature extraction
X = df_train.loc[:, df_train.columns != 'label_pred']
Y= df_train.loc[:,'label_pred']

model = LogisticRegression()
rfe = RFE(model, 15)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_),X.columns)
print("Feature Ranking: %s" % (fit.ranking_))

## Here we have top 15 features as 
## cycle,s1,s3,s4,s7,s8,s9,s11,s12,s13,s14,s17,s18,s20,ttf

# ********************* Principal component Analysis **********************
features = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21','ttf']
# Separating out the features
x = df_train.loc[:, features].values
# Separating out the predictor
y = df_train.loc[:,['label_pred']].values
# Standardise the data to have an equal standard deviation of 1 for all
scaled_data = preprocessing.scale(x)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
per_var = np.round(pca.explained_variance_ratio_*100, decimals =1 )
labels = ['PC' +str(x) for x in range(1,len(per_var)+1)]

## Plotting the scree plot to explain the maximum variance 
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('scree Plot')
plt.bar(x = range(1, len(per_var)+1), height = per_var,tick_label = labels)
plt.show()
# All of the variance is being explained with PC1 itself 
loading_scores = pd.Series(pca.components_[0], index=X.columns)
sorted_loading_scores = loading_scores.abs().sort_values(ascending = False)
top_15_Features = sorted_loading_scores[0:15]

# Features Selected according to PCA which explain the maximum variance 
# s11,s12,s4,s7,s15,s21,s20,s8,s13,s2,17,ttf,s3,cycle

#**** Select K Best Feautures based on chi-square statistical tests *********************
# Separating out the features
# Removed settings 1 and 2 since they have negative values 
#for chi2 statistical test the pre condition is that there should be no negative values.
# I also safely assume that settings will unlikely turn to be among best features since none of  other methods have chosen settings
ftrs = ['id','cycle','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21','ttf']
Train_features = df_train.loc[:, ftrs]
# Separating out the predictor
Train_Target = df_train.loc[:,['label_pred']]
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=15)
fit = bestfeatures.fit(Train_features,Train_Target)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Train_features.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)

#naming the dataframe columns
featureScores.columns = ['Specs','Score']
#print 10 best features  
print(featureScores.nlargest(15,'Score'))  

# Features selected according to K best statistical scores
# ttf,cycle,s4,s9,s3,s19,id,s17,s11,s7,s12,s20,s21,s2,s15


##************************* Using Borutapy*************************
X1 = df_train.drop(['label_pred'], axis=1).values
Y1 = df_train['label_pred'].values

rfc = RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight='balanced', max_depth=6)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
#start_time = timer(None)
boruta_selector.fit(X1,Y1)

# number of selected features
print ('\n Number of selected features:')
print (boruta_selector.n_features_)

feature_df = pd.DataFrame(df_train.drop(['label_pred'], axis=1).columns.tolist(), columns=['features'])
feature_df['rank']=boruta_selector.ranking_
feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)
print ('\n Top %d features:' % boruta_selector.n_features_)
print (feature_df.head(boruta_selector.n_features_))
feature_df.to_csv('boruta-feature-ranking.csv', index=False)

# check ranking of features
print ('\n Feature ranking:')
print (boruta_selector.ranking_)

# Features Selected using Boruta
# Around 17 features are selected 
#id,cycle,s2,s3,s4,s7,s8,s9,s11,s12,s13,14,s15,s17,s20,ttf

# *************Feature Importance with Random Forest****************8

# fit an Extra Trees model to the data
df_tr_cols = df_train.loc[:, df_train.columns != 'label_pred']
df_tr_tgt= df_train.loc[:,'label_pred']
model = ExtraTreesClassifier()
model.fit(df_tr_cols,df_tr_tgt)
# display the relative importance of each attribute
print(model.feature_importances_)
# Sorting the features according to their importance
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = df_tr_cols.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)

# Features Selected based on feature importance
# ttf,s20,s11,s7,s2,s15,s14,s9,s8,s4,s13,s12,s17,cycle,s21
