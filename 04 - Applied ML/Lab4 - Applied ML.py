
# coding: utf-8

# # Applied Machine learning !

# In this lab we will study a soccer dataset and apply Machine Learning to it. We will try to predict the skin color of a player given some of it's information

# ### Some useful imports

# In[ ]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ### First we import the dataset and study it :

# In[ ]:

df = pd.read_csv("./CrowdstormingDataJuly1st.csv")


# How much data do we have ?

# In[ ]:

df.shape


# Let's look at our columns

# In[ ]:

# Taken form http://nbviewer.jupyter.org/github/mathewzilla/redcard/blob/master/Crowdstorming_visualisation.ipynb 
df.head().ix[:10,0:13]


# In[ ]:

# Taken form http://nbviewer.jupyter.org/github/mathewzilla/redcard/blob/master/Crowdstorming_visualisation.ipynb 
df.head().ix[:10,13:28]


# In[ ]:

df[df.rater1 == df.rater2].shape


# In[ ]:

df["rater2"].unique()


# Thanks to the previous work of Mat Evans and Tom Stafford and their team (can be found here : http://nbviewer.jupyter.org/github/mathewzilla/redcard/blob/master/Crowdstorming_visualisation.ipynb )
# 
# we know some of the specificities of the data :

# - We learn that the data is organized in a player-referee dyad. This means each row has all the interaction with one player and a referee. For their analysis they decided to separate this dataset for each interaction between a player and a referee. For our purposes we will rather group by player short because we don't really care for referee information.
# 
# - The data about skin color is not consistent between the two raters and the picture is missing. For the picture we will just remove the data with no picture. As we need a way to attribute skin color to each player we will have to use the two raters. For this there are different ways we could go at it :
#     - We could do a mean of the two raters
#     - We could keep only the dyads which have the same score for r1 and r2
#     - We could train on both raters and compare the result (seems a bit repetitive but why not)
#     
# - The raters data in the set [0, 0.25, 0.50, 0.75, 1] to classify "correctly" there are 3 possibilities :
#     - The first one is to have a class for each number
#     - The second is to have two classes with a cut at 0.5 (we have to define which class contain 0.5)
#     - The third is to have 3 classes : $<0.5$, $0.5$, $>0.5$
# 
# - Finally we learn that there are dyads which are not really part of the 2012-2013 data but from past matches in the carrer of the player. Here we have a choice, either we remove the data because it's not really part of the dataset or we let it be as it could be more information.

# Here is how we think of organizing this homework :
# - First we will clean a bit the data
# - Then we will aggregate the data per player
# - Then we will for each point made above with multiple proposition see which one is the best, if multiple ones seem good we will keep them and test them with cross validation.
# - Finally we will find the best result we can with all possible datasets with a random forest classifier.

# ## Removing non valid dyads from player past history

# For a more detailed explanation of how we can show the fact that there are dyads from before 2012-2013 you can look here : http://nbviewer.jupyter.org/github/mathewzilla/redcard/blob/master/Crowdstorming_visualisation.ipynb

# To remove those dyad, the logic is to say that if a referee has participated in at least one match in the league he should have at least 22 appearances in the dataset.

# The folowing code ressembles (as we use the same principles) the work previously mentioned. But as we will see some differences I will try and explain them as we go through the code to show we aren't doing anything wrong.

# In[ ]:

referee_b = df["refNum"].unique().shape[0]
print("number of unique referee before cleaning : ", referee_b) # here no differences


# In[ ]:

dyads_b = df.shape[0]
games_b = df.games.sum()
print("number of dyads before cleaning : ", dyads_b)
print("number of matches before cleaning : ", games_b) # same value as in the other notebook 
                                                     # (we will use it later as reference)


# This part shows the value that the others get :

# In[ ]:

apearances_tot = df[["refNum", "games"]].groupby("refNum").sum()


# In[ ]:

apearances_sup21_tot = apearances_tot[apearances_tot.games > 21]
apearances_sup21_tot.count() # same number of remaining referee as in the other notebook


# In[ ]:

df_sup21_tot = df[df["refNum"].isin(apearances_sup21_tot.index)]


# In[ ]:

referee_a_tot = df_sup21_tot["refNum"].unique().shape[0]
print("number of unique referee after removing : ", referee_a_tot)


# In[ ]:

dyads_a_tot = df_sup21_tot.shape[0]
games_a_tot = df_sup21_tot.games.sum()
print("number of dyads after removing : ", dyads_a_tot)
print("number of matches after removing : ", games_a_tot) # same number as in the other notebook 


# Now as our data is organized differently we use another method

# Here is what will change most of the calculations result

# In[ ]:

apearances_once_player = df.refNum.value_counts()
len(apearances_once_player)


# This is the number of apearances with a slight twist, we only count one apearance per player whatever the number of match he played. For us this value is better to use in this case as it relates more to the >21 cut. Before we counted for a single referee a number of matches with the same player which makes no sense : in a match, a referee has 22 distinct players.

# In[ ]:

apearances_sup21_once_player = apearances_once_player[apearances_once_player > 21]
len(apearances_sup21_once_player) ## a bit less than before sadly


# In[ ]:

df_sup21_once_player = df[df["refNum"].isin(apearances_sup21_once_player.index.values)]


# In[ ]:

referee_a_once_player = df_sup21_once_player["refNum"].unique().shape[0]
print("number of unique referee after removing : ", referee_a_once_player)


# In[ ]:

dyads_a_once_player = df_sup21_once_player.shape[0]
games_a_once_player = df_sup21_once_player.games.sum()
print("number of dyads after removing : ", dyads_a_once_player)
print("number of matches after removing : ", games_a_once_player) # a bit lower than before again


# Let's show how much data we lose

# In[ ]:

print("loss of games with their method :", games_a_tot / games_b)
print("loss of dyads with their method :", dyads_a_tot / dyads_b)
print("loss of refs with their method : ", referee_a_tot / referee_b)


# In[ ]:

print("loss of games with our method :", games_a_once_player / games_b)
print("loss of dyads with our method :", dyads_a_once_player / dyads_b)
print("loss of refs with our method : ", referee_a_once_player / referee_b)


# ### Here we show some graphes to show the difference between the two methods.

# Graph of occurences (source from http://nbviewer.jupyter.org/github/mathewzilla/redcard/blob/master/Crowdstorming_visualisation.ipynb)
# 
# We thought of using their graphs to have a good comparaison between their work and ours

# In[ ]:

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
axes.hist(df[["refNum", "games"]].groupby("refNum").sum().games.tolist(),referee_b-11)
axes.set_xscale('symlog') # symetric log scale 
plt.xlim([1,10000])
axes.set_yscale('symlog') 
plt.ylim([0,1000])
axes.set_title("Referee occurance following our cull, log scaled")
axes.set_xlabel('log (number of occurances)')
axes.set_ylabel('log (frequency)')


# In[ ]:

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
axes.hist(df_sup21_tot[["refNum", "games"]].groupby("refNum").sum().games.tolist(),referee_b-11)
axes.set_xscale('symlog') # symetric log scale 
plt.xlim([1,10000])
axes.set_yscale('symlog') 
plt.ylim([0,1000])
axes.set_title("Referee occurance following our cull, log scaled")
axes.set_xlabel('log (number of occurances)')
axes.set_ylabel('log (frequency)')


# In[ ]:

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
axes.hist(df_sup21_once_player[["refNum", "games"]].groupby("refNum").sum().games.tolist(),referee_b-11)
axes.set_xscale('symlog') # symetric log scale 
plt.xlim([1,10000])
axes.set_yscale('symlog') 
plt.ylim([0,1000])
axes.set_title("Referee occurance following our cull, log scaled")
axes.set_xlabel('log (number of occurances)')
axes.set_ylabel('log (frequency)')


# We can clearly see here that we only remove the lowest occurences of the previous graph

# In[ ]:

fig, ax = plt.subplots(1,1,figsize=(12, 4))
x = df.refCountry.value_counts()
lines = ax.plot(x.values,marker='.',ms=20)

y = x.index.tolist() 

ax.set_title('Referee nationality by number of dyads')
ax.set_xlabel('Country number (ordered by frequency)')
ax.set_ylabel('Frequency of dyads')
ax.set_xlim([-3,160]) # a hack so we can see the first point most clearly


# In[ ]:

fig, ax = plt.subplots(1,1,figsize=(12, 4))
x = df_sup21_tot.refCountry.value_counts()
lines = ax.plot(x.values,marker='.',ms=20)

y = x.index.tolist() 

ax.set_title('Referee nationality by number of dyads')
ax.set_xlabel('Country number (ordered by frequency)')
ax.set_ylabel('Frequency of dyads')
ax.set_xlim([-3,160]) # a hack so we can see the first point most clearly


# In[ ]:

# Histogram of country frequency. 
fig, ax = plt.subplots(1,1,figsize=(12, 4))
x = df_sup21_once_player.refCountry.value_counts()
lines = ax.plot(x.values,marker='.',ms=20)

y = x.index.tolist()

ax.set_title('Referee nationality by number of dyads')
ax.set_xlabel('Country number (ordered by frequency)')
ax.set_ylabel('Frequency of dyads')
ax.set_xlim([-3,160]) # a hack so we can see the first point most clearly


# Here we see that our method reduces the number of total countries but doesn't change the frequency of dyads with the most represented countries compared to the previous graph

# ## Removing non usable data for our ML algorithm

# We know sometimes there is no images for a player and therefor no skin color rating. Therefore we remove them.

# In[ ]:

df_with_pic = df_sup21_once_player[df_sup21_once_player["photoID"].notnull()]


# In[ ]:

df_sup21_once_player.shape[0]


# In[ ]:

df_with_pic.shape[0]


# Should be all clean now.

# In[ ]:

dfc = df_with_pic


# ## Aggregating the data by player

# In[ ]:

dfc.describe()


# Let's look at a single players dyad to have an idea of what it looks like

# In[ ]:

groups = dfc.groupby("playerShort")


# In[ ]:

lucas = groups.get_group("lucas-wilchez")


# In[ ]:

# Taken form http://nbviewer.jupyter.org/github/mathewzilla/redcard/blob/master/Crowdstorming_visualisation.ipynb 
lucas.head().ix[:,:13]


# In[ ]:

# Taken form http://nbviewer.jupyter.org/github/mathewzilla/redcard/blob/master/Crowdstorming_visualisation.ipynb 
lucas.head().ix[:,13:]


# First we will only keep the columns that seem relevant to us and remove the ones we think are not important :
# - player name which is a repetition of the playershort
# - birthday as it should not be relevant, completely independent (that would be weird al least :))
# - refNum, refCountry, Alpha3 which are not easily usable as an aggregated feature (a solution to use this part of the data will be detailed later)
# - nIAT and nExp really not relevant here 
#     we could do a weihted mean with the "mean" and "se" columns but it seemed a bit exagerated (could be a futur improvement ?)
#     
# Why we kept other columns is explained below with the aggregation operation.

# In[ ]:

# We only keep what we think are relevant columns.
df_filtered = dfc[["playerShort","club", "leagueCountry", "height", "weight", "position", "games", "victories", 
                 "ties", "defeats", "goals", "yellowCards", "yellowReds", "redCards",
                 "rater1", "rater2", "meanIAT", "seIAT", "meanExp", "seExp"]]


# In[ ]:

df_grouped = df_filtered.groupby("playerShort").agg({
        "club": lambda x: x.unique()[0],
        "leagueCountry": lambda x: x.unique()[0],
        "height": np.max,
        "weight": np.max,
        "position": lambda x: x.unique()[0],
        "games": np.sum,
        "victories": np.sum,
        "ties": np.sum,
        "defeats": np.sum,
        "goals": np.sum,
        "yellowCards": np.sum,
        "yellowReds": np.sum,
        "redCards": np.sum,
        "rater1": np.max, # never changes so we can take either min, max or mean 
        "rater2": np.max, # same here (we used this to test that nothing changed : [np.min, np.max, np,mean])
        "meanIAT": np.mean, # Here doing the mean seems a bit confusing but it will give an 
        "seIAT": np.mean,   # indicatiion whether the player could have been mistreated in 
                            # some of his matches or never.
        "meanExp": np.mean, # Same here
        "seExp": np.mean    # We could have applied pooled variance (will see later) : https://en.wikipedia.org/wiki/Pooled_variance
    })


# In[ ]:

# We used this to check wether min and max rating change for each player (which was not the case)
#df_grouped[df_grouped["rater1", "amin"] != df_grouped["rater1", "amax"]]


# In[ ]:

# We used this to test if some players had multiple clubs, league country or position. Which was not the case.
#print(df_grouped["club"].apply(lambda x: x[0].shape).unique())
#print(df_grouped["leagueCountry"].apply(lambda x: x[0].shape).unique())
#print(df_grouped["position"].apply(lambda x: x[0].shape).unique())


# In[ ]:

df_grouped.head()


# In[ ]:

len(df_grouped)


# Thanks to the description of the data in DATA.md we know there should be 1586 players with pictures.
# With this value we can validate the fact that we lost minimal data and have most of the players with a picture.

# ## Preparing Data

# As a last step before starting to do machine learning we need to reformat the data and seperate it

# First we need to make rows which contain strings in integers (club, position, leagueCountry)

# In[ ]:

from sklearn import preprocessing

df_grouped["club"] = df_grouped["club"].astype(np.str)
df_grouped["position"] = df_grouped["position"].astype(np.str)
df_grouped["leagueCountry"] = df_grouped["leagueCountry"].astype(np.str)

def encodeLabels(col, df):
    le = preprocessing.LabelEncoder()
    le.fit(df[col].unique())
    df[col] = le.transform(df[col])

encodeLabels("club", df_grouped)
encodeLabels("position", df_grouped)
encodeLabels("leagueCountry", df_grouped)


# Now we can create the futur x and y for training

# In[ ]:

y_possible = df_grouped[["rater1","rater2"]]
y_possible.head()


# In[ ]:

x = df_grouped.drop(y_possible, axis=1)
x.head()


# In[ ]:

def prepFeature(feature) :
    nans = True in x[feature].isnull().unique()
    
    if nans:
        print("mean replacement of nans")
        x[feature] = x[feature].fillna(int(x[feature].mean()))
    
    f, ax = plt.subplots(1, 1)
    ax.set_title("histogram of feature " + feature)
    ax.hist(x[feature].values)


# In[ ]:

prepFeature("redCards")


# In[ ]:

prepFeature("games")


# In[ ]:

prepFeature("defeats")


# In[ ]:

prepFeature("height")


# In[ ]:

prepFeature("meanExp")


# In[ ]:

prepFeature("goals")


# In[ ]:

prepFeature("yellowCards")


# In[ ]:

prepFeature("yellowReds")


# In[ ]:

prepFeature("club")


# In[ ]:

prepFeature("weight")


# In[ ]:

prepFeature("seIAT")


# In[ ]:

prepFeature("ties")


# In[ ]:

prepFeature("leagueCountry")


# In[ ]:

prepFeature("meanIAT")


# In[ ]:

prepFeature("victories")


# In[ ]:

prepFeature("seExp")


# In[ ]:

prepFeature("position")


# ## Naive machine learning

# In[ ]:

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics


# In[ ]:

rfc = RFC(n_estimators=10, n_jobs=-1, class_weight=None)


# Let's build y (in a naive fashion for now)

# In[ ]:

y = (y_possible['rater1'] + y_possible['rater2'] / 2 < 0.5).values


# In[ ]:

rfc.fit(x, y)


# In[ ]:

y_pred = rfc.predict(x)


# In[ ]:

print(metrics.mean_absolute_error(y, y_pred))
print(metrics.accuracy_score(y, y_pred))


# ### Ok so we're done here ! 99% seems reasonable, goodbye !

#  ...

#  ...

#  ...

#  ...

# ## Ok just kidding let's get serious with this !

# First let's really test the naive approach and see how it could perform in a "real" situation.

# let's first split the dataset into a training and testing set. This seems to be generally a good practice in machine learning :).

# In[ ]:

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)


# In[ ]:

rfc = RFC(n_estimators=10, n_jobs=-1, class_weight=None)


# In[ ]:

rfc.fit(x_train, y_train)


# In[ ]:

y_pred = rfc.predict(x_test)


# In[ ]:

print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))


# Ok that's kind of disapointing... (but not so much suprising). A better way to show the error is cross validation.

# In[ ]:

#rfc = RFC(n_estimators=10, n_jobs=-1, class_weight=None)


# In[ ]:

# Cross validation 10-Fold (for now) with accuracy scoring
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(rfc, x, y, cv=10, scoring='accuracy')


# In[ ]:

def show_score(scores):
    print(scores)
    print("--------------------------")
    print("mean :", np.mean(scores))
    print("min :", np.min(scores))
    print("max :", np.max(scores))

show_score(scores)


# We can see again that the result is not pretty, the mean result we have is ~70% which is better than perdicting that all players are light skinned

# proportions of classes for the mean rating (considering 1 -> $mean \leq 0.5$)

# In[ ]:

# Proportion of light and dark skinned players
prop_1 = np.sum(y) / len(y)
prop_0 = 1 - prop_1
print("proportion of ones :", prop_1)
print("proportion of zeroes :", prop_0)


# Let's try to use some well known ML methods to understand what's going wrong.

# Let's look at what the confusion matrix has to say.

# In[ ]:

confusion_mx = metrics.confusion_matrix(y_test, y_pred)
TP = confusion_mx[1, 1]
TN = confusion_mx[0, 0]
FP = confusion_mx[0, 1]
FN = confusion_mx[1, 0]


# In[ ]:

confusion_mx


# |total : 317| pred : 0 |  pred : 1  |
# |---|----|-----|
# | actual : 0 | TN = ~30 | FP = ~55 |
# | actual : 1 | FN = ~26 | TP = ~200 |

# We can see here that we are good at predicting ones, but our predictions of 0 are all over the place.

# There is an easy way to show this : the **Specificity** (or how correct is the classifier with 0 values)

# In[ ]:

specificity = TN / float(TN + FP)
print("Specificity :", specificity)


# We can compare it to **Sensitivity** (or true positive rate)

# In[ ]:

sensitivity = TP / float(TP + FN)
print("sensitivity :", sensitivity)


# Which is much better. 
# 
# PS : all these methods are taken from the course and the code was copied from this notebook :
# 
# http://nbviewer.jupyter.org/github/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb

# So, how are we going to do a better job ?

# The first thing we realize is that there is a way to indicate to the random forest classifier the fact that there is a disparity within the data.

# In[ ]:

class_weights = {
    1 : prop_1*10,
    0 : prop_0*10
}


# In[ ]:

rfc = RFC(n_estimators=10, n_jobs=-1, class_weight=class_weights)


# We will use a function that prints out most of the information we used above to test our new rfc 

# In[ ]:

from helpers import test_rfc
test_rfc(rfc, x, y)


# Ok, to bad it's not better than before in terms of accuracy, eventhough we have a better score with specificity.

# Let's try something else : changing the classification threshold

# We think this will help the **specificity** get higher.

# let's retrain our data with our new rfc (with weights)

# In[ ]:

rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)


# getting the probability of ones of the classifier

# In[ ]:

y_pred_prob = rfc.predict_proba(x_test)[:, 1]


# Separating the probability of true and false values

# In[ ]:

y_pred_prob1 = [x[1] for x in zip(y_test, y_pred_prob) if x[0]]
y_pred_prob0 = [x[1] for x in zip(y_test, y_pred_prob) if not x[0]]


# The following graph was inspired by this video :
# 
# https://www.youtube.com/watch?v=OAl6eAyP-yo
# 
# It shows in blue the probability given to the true 0 values and in red the probability of true 1 values.

# In[ ]:

# histogram of predicted probabilities
plt.hist(y_pred_prob1, bins=10, alpha=0.6, color="red")
plt.hist(y_pred_prob0, bins=10, alpha=0.6, color="blue")
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probabilities')
plt.ylabel('Frequency')


# With this graph we realized that it's going to be very difficult to have a good and fair (in terms of specificity and sensitivity) 

# To verify this fact we are going to use the ROC curve and the AUC (Area Under the Curve) metric

# In[ ]:

# Code copied entirely from 
# http://nbviewer.jupyter.org/github/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb
# There wasn't really another way to show that though :)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for our classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[ ]:

# calculate cross-validated AUC score
AUC_mean = cross_val_score(rfc, x, y, cv=10, scoring='roc_auc').mean()
print("AUC score :", AUC_mean)


# Ok, so what does this all mean.

# First the ROC curve shows us what are the best compromises we can make between Specificity and Sensitivity (be aware that here specificity is inversed), with that you could chose precisely what you want the classifier to be compromising.

# For the AUC score it shows a score of our model compared to a random sampling of 1's and 0's. (PS. the random sample is weighted of course)

# Ok now we have all the tools to try and accuratly validate a method

# ## Modifying the model

# ### Some tips and tricks from the lab session.

# In[ ]:

# print the first 10 predicted responses
logreg.predict(X_test)[0:10]

# store the predicted probabilities for class 1
y_pred_prob = logreg.predict_proba(X_test)[:, 1]


# In[ ]:

# STEP 1: split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# In[ ]:

# simulate splitting a dataset of 25 observations into 5 folds
from sklearn.cross_validation import KFold
kf = KFold(25, n_folds=5, shuffle=False)

# print the contents of each training and testing set
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], data[1]))


# In[ ]:

from sklearn.cross_validation import cross_val_score
# 10-fold cross-validation
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')


# In[ ]:

from sklearn.grid_search import GridSearchCV

# define the parameter values that should be searched
k_range = list(range(1, 31))

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)

# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

#You can set n_jobs = -1 to run computations in parallel (if supported by your computer and OS

# fit the grid with data
grid.fit(X, y)

# view the complete results (list of named tuples)
grid.grid_scores_

# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)

# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

# shortcut: GridSearchCV automatically refits the best model using all of the data
grid.predict([[3, 5, 4, 2]])

# n_iter controls the number of searches
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
rand.fit(X, y)
rand.grid_scores_


# In[ ]:

from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))

# calculate RMSE using scikit-learn
print(np.sqrt(metrics.mean_squared_error(true, pred)))

print(metrics.accuracy_score(y_test, y_pred))

# IMPORTANT: first argument is true values, second argument is predicted values
print(metrics.confusion_matrix(y_test, y_pred_class))

# Sensitivity : Recall or True Positive Rate
print(TP / float(TP + FN))
print(metrics.recall_score(y_test, y_pred_class))

# Specificity score (how precise is the classifier for positive value)
print(TN / float(TN + FP))

# Flase positive Rate 
print(FP / float(TN + FP))

# Precision, true positive rate.
print(TP / float(TP + FP))
print(metrics.precision_score(y_test, y_pred_class))


# In[ ]:

# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize
y_pred_class = binarize([y_pred_prob], 0.3)[0]

# IMPORTANT: first argument is true values, second argument is predicted probabilities
# Plot of the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

