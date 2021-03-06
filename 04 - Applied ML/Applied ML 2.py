
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from sklearn import metrics


# In[2]:

df = pd.read_csv("./CrowdstormingDataJuly1st.csv")


# In[3]:

from helpers import clean_data, group_data, prep_ML, normalize
dfc = clean_data(df)
dfg = group_data(dfc)
X_p,y_possible = prep_ML(dfg)
X = normalize(X_p, None)
# We use our best version of the previous part.
label_true = ((y_possible['rater1'] + y_possible['rater2']) / 2 <= 0.5).values


# First let's try the k-means algorithm with 2 clusters and print out the silhouette score:

# In[4]:

from sklearn.cluster import KMeans


# FYI: n_jobs = 1 because the parallel version of k-means doesn't work on OSX.

# In[5]:

km = KMeans(n_clusters=2, max_iter=600, init="k-means++", n_jobs=1 )


# In[6]:

km.fit(X)


# In[7]:

labels = km.labels_
s1 = metrics.silhouette_score(X, labels, metric='euclidean')
print("s1 =", s1)


# In[8]:

def test_km(X):
    km = KMeans(n_clusters=2, max_iter=600, init="k-means++", n_jobs=1 )
    km.fit(X)
    labels = km.labels_
    return metrics.silhouette_score(X, labels, metric='euclidean')


# We know that the most valuable feature is the seIAT therefore we will remove it (just for fun)

# In[9]:

from helpers import compute_feature_importance_rfc
from sklearn.ensemble import RandomForestClassifier as RFC

# From previous exercice

prop1 = np.sum(label_true) / len(label_true)
prop0 = 1 - prop1
class_weights = {
    0 : prop0,
    1 : prop1
}

best_results = {'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 25}

rfc = RFC(max_depth=best_results["max_depth"], max_features=best_results["max_features"], n_estimators=best_results["n_estimators"], n_jobs=-1, class_weight=class_weights)
compute_feature_importance_rfc(rfc, X, label_true)


# In[10]:

X2 = X.drop("seIAT", axis=1)
s2 = test_km(X2)
print("s2 =", s2)
print("s2 - s1 =", s2 - s1)


# We have a better classification... that's fun !

# Ok now let's remove a second feature, we will follow the same intuition as before and remove the second best feature

# In[11]:

# we know from before that the mean_Exp and goal information is at position 8, and 7 respectively in the X array
X3 = X.drop(["meanExp", "seIAT"], axis=1)
s3 = test_km(X3)
print("s3 =", s3)
print("s3 - s1 =", s3 - s1)
print("s3 - s2 =", s3 - s2)


# There is an even better improvement

# In[12]:

# we now remove seExp in addition to the other 2 (3rd best feature)
X4 = X.drop(["meanExp", "seExp", "seIAT", "meanIAT"], axis=1)
s4 = test_km(X4)
print("s4 =", s4)
print("s4 - s1 =", s4 - s1)
print("s4 - s2 =", s4 - s2)
print("s4 - s3 =", s4 - s3)


# To see whether the clustering is close to a dark/light separation we will compute in addition to the silhouette the adjusted mutual info score 

# In[13]:

km = KMeans(n_clusters=2, max_iter=300, init="k-means++", n_jobs=1 )

def scoring_complete(X):
    km.fit(X)
    labels = km.labels_
    print("silhouette score :", metrics.silhouette_score(X, labels, metric='euclidean'))
    print("closeness to true label score :", metrics.adjusted_mutual_info_score(label_true, labels))

scoring_complete(X)


# In[14]:

scoring_complete(X2)


# Here we see that eventhough the silhouette score is better our label accuracy is the same as before.

# Other examples :

# In[15]:

scoring_complete(X3)


# In[16]:

scoring_complete(X4)


# Now let's remove the worst features for example the red / yellow / redYellow / cards (not all are the absolute worsts but they all were in the < 0.05 importance in the previous exercice.)

# In[17]:

X5 =  X.drop(["yellowReds", "redCards", "yellowCards"], axis=1)
scoring_complete(X5)


# We have good results ! The silhouette is better, and closeness is much better

# Let's now remove all the features that have the worst feature importance until our closeness score drops significantly

# In[18]:

compute_feature_importance_rfc(rfc, X5, label_true)


# In[19]:

X6 =  X5.drop(["position", "leagueCountry", "height"], axis=1)
scoring_complete(X6)


# In[20]:

compute_feature_importance_rfc(rfc,X6, label_true)


# In[21]:

X7 =  X6.drop(["defeats"], axis=1)
scoring_complete(X7)


# In[22]:

compute_feature_importance_rfc(rfc, X7, label_true)


# In[23]:

X8 =  X7.drop(["weight"], axis=1)
scoring_complete(X8)


# In[24]:

compute_feature_importance_rfc(rfc, X8, label_true)


# In[25]:

X9 =  X8.drop(["ties"], axis=1)
scoring_complete(X9)


# In[26]:

compute_feature_importance_rfc(rfc, X9, label_true)


# In[27]:

X10 =  X9.drop(["goals"], axis=1)
scoring_complete(X10)


# In[28]:

compute_feature_importance_rfc(rfc, X10, label_true)


# In[29]:

X11 =  X10.drop(["club"], axis=1)
scoring_complete(X11)


# In[30]:

compute_feature_importance_rfc(rfc, X11, label_true)


# In[31]:

X12 =  X11.drop(["birthday"], axis=1)
scoring_complete(X12)


# In[32]:

compute_feature_importance_rfc(rfc, X12, label_true)


# In[33]:

X13 =  X12.drop(["victories"], axis=1)
scoring_complete(X13)


# Removing more gives worse results or equal results.
