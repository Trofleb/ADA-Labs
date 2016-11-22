
# coding: utf-8

# In[ ]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from sklearn import metrics


# In[ ]:

df = pd.read_csv("./CrowdstormingDataJuly1st.csv")


# In[ ]:

from helpers import clean_data, group_data, prep_ML, normalize
dfc = clean_data(df)
dfg = group_data(dfc)
X_p,y_possible = prep_ML(dfg)
X = normalize(X_p, None)
# We use our best version of the previous part.
label_true = ((y_possible['rater1'] + y_possible['rater2']) / 2 <= 0.5).values


# First let's try the k-means algorithm with 2 clusters and print out the silhouette score:

# In[ ]:

from sklearn.cluster import KMeans


# FYI: n_jobs = 1 because the parallel version of k-means doesn't work on OSX.

# In[ ]:

km = KMeans(n_clusters=2, max_iter=600, init="k-means++", n_jobs=1 )


# In[ ]:

km.fit(X)


# In[ ]:

labels = km.labels_
s1 = metrics.silhouette_score(X, labels, metric='euclidean')
print("s1 =", s1)


# In[ ]:

def test_km(X):
    km = KMeans(n_clusters=2, max_iter=600, init="k-means++", n_jobs=1 )
    km.fit(X)
    labels = km.labels_
    return metrics.silhouette_score(X, labels, metric='euclidean')


# We know that the most valuable feature is the mean_Exp therefore we will remove it (just for fun)

# In[ ]:

from helpers import compute_feature_importance_rfc

compute_feature_importance_rfc(X, label_true)


# In[ ]:

X2 = X.drop("meanExp", axis=1)
s2 = test_km(X2)
print("s2 =", s2)
print("s2 - s1 =", s2 - s1)


# We have a better result... that's fun !

# Ok now let's remove a second feature, we will follow the same intuition as before and remove the second best feature, the club information.

# In[ ]:

# we know from before that the mean_Exp and goal information is at position 8, and 7 respectively in the X array
X3 = X.drop(["meanExp", "goals"], axis=1)
s3 = test_km(X3)
print("s3 =", s3)
print("s3 - s1 =", s3 - s1)
print("s3 - s2 =", s3 - s2)


# There is an even better improvement

# In[ ]:

# we now remove seExp in addition to the other 2 (3rd best feature)
X4 = X.drop(["meanExp", "goals", "seIAT"], axis=1)
s4 = test_km(X4)
print("s4 =", s4)
print("s4 - s1 =", s4 - s1)
print("s4 - s2 =", s4 - s2)
print("s4 - s3 =", s4 - s3)


# To see whether the clustering is close to a dark/light separation we will compute in addition to the silhouerre the accuracy score 

# In[ ]:

def scoring_complete(X):
    km = KMeans(n_clusters=2, max_iter=300, init="k-means++", n_jobs=1 )
    km.fit(X)
    labels = km.labels_
    print("silhouette score :", metrics.silhouette_score(X, labels, metric='euclidean'))
    print("closeness to true label score :", metrics.adjusted_mutual_info_score(label_true, labels))

scoring_complete(X)


# We have a very low closeness score but at least it's positive.

# In[ ]:

scoring_complete(X2)


# Here we see that eventhough the silhouette score is better our label accuracy is the same as before.

# Other example :

# In[ ]:

scoring_complete(X3)


# In[ ]:

scoring_complete(X4)


# Clearly here we are changinging the biais of the clustering algorithm (with respect to our "true" labels) **TODO : verify this !**

# Now let's remove the worse features for example the red / yellow / redYellow / cards (not all are the absolute worst but they were all in the < 0.05 importance in the previous exercice.)
# 
# rows : 8, 10, 11

# In[ ]:

X5 =  X.drop(["yellowReds", "redCards", "yellowCards"], axis=1)
scoring_complete(X5)


# We have good results ! the silhouette is good and closeness has improved.

# Let's now remove all the features that have the worst feature importance until our closeness score drops significantly

# In[ ]:

compute_feature_importance_rfc(X5, label_true)


# In[ ]:

X6 =  X.drop(["yellowReds", "redCards", "yellowCards", "position", "leagueCountry", "height"], axis=1)
scoring_complete(X6)


# In[ ]:

compute_feature_importance_rfc(X6, label_true)


# In[ ]:

X7 =  X.drop(["yellowReds", "redCards", "yellowCards", "position", "leagueCountry", "height", "weight"], axis=1)
scoring_complete(X7)


# In[ ]:

compute_feature_importance_rfc(X7, label_true)


# In[ ]:

X8 =  X.drop(["yellowReds", "redCards", "yellowCards", "position", "leagueCountry", "height", "weight", "defeats"], axis=1)
scoring_complete(X8)


# In[ ]:

compute_feature_importance_rfc(X8, label_true)


# In[ ]:

X9 =  X.drop(["yellowReds", "redCards", "yellowCards", "position", "leagueCountry", "height", "weight", "defeats", "ties"], axis=1)
scoring_complete(X9)


# We stop here as we have the worst result.

# In[ ]:

plt.scatter(X8["club"], X8["meanExp"], c=labels)


# In[ ]:



