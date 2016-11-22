import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

from sklearn.ensemble import RandomForestClassifier as RFC


def clean_data(df):
    apearances_once_player = df.refNum.value_counts()
    apearances_sup21_once_player = apearances_once_player[apearances_once_player > 21]
    df_sup21_once_player = df[df["refNum"].isin(apearances_sup21_once_player.index.values)]

    df_with_pic = df_sup21_once_player[df_sup21_once_player["photoID"].notnull()]
    
    return df_with_pic

def group_data(df):
    # We only keep what we think are relevant columns.
    df_filtered = df[["playerShort","club", "leagueCountry", "height", "weight", "position", "games", "victories", 
                 "ties", "defeats", "goals", "yellowCards", "yellowReds", "redCards",
                 "rater1", "rater2", "meanIAT", "seIAT", "meanExp", "seExp"]]
    
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
    
    return df_grouped
    
def encodeLabels(col, df):
    le = LabelEncoder()
    le.fit(df[col].unique())
    df[col] = le.transform(df[col])

def prep_ML(df):

    df["club"] = df["club"].astype(np.str)
    df["position"] = df["position"].astype(np.str)
    df["leagueCountry"] = df["leagueCountry"].astype(np.str)

    encodeLabels("club", df)
    encodeLabels("position", df)
    encodeLabels("leagueCountry", df)
    
    for feature, col in df.iteritems():
        has_nan = True in col.isnull().unique()
        if has_nan:
            df[feature] = col.fillna(int(col.mean()))

    y_possible = df[["rater1","rater2"]]
    x = df.drop(y_possible, axis=1)
    
    return x, y_possible

def normalize(X, y):
    normalizer = Normalizer()
    for feature, col in X.iteritems():
        has_nan = True in col.isnull().unique()
        if has_nan:
            X[feature] = normalizer.fit_transform(col, y)
    
    return X

def show_score(scores):
    print("Cross validation scores")
    print(scores)
    print("--------------------------")
    print("cross mean :", np.mean(scores))
    print("cross min :", np.min(scores))
    print("cross max :", np.max(scores))

def specificity(y, y_pred, **kwargs):
    confusion_mx = metrics.confusion_matrix(y, y_pred)
    TP = confusion_mx[1, 1]
    TN = confusion_mx[0, 0]
    FP = confusion_mx[0, 1]
    FN = confusion_mx[1, 0]
    
    specificity = TN / float(TN + FP)
    return specificity

specificity_scorer = metrics.make_scorer(specificity)
    
def test_rfc(rfc, x, y):
    
    # Prep of training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    
    # Do prediction for test values
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    
    # Cross validation 10-Fold (for now) with accuracy scoring
    scores = cross_val_score(rfc, x, y, cv=5, scoring='accuracy')
    show_score(scores)
    
    # Let's compute the convolution matrix
    confusion_mx = metrics.confusion_matrix(y_test, y_pred)
    TP = confusion_mx[1, 1]
    TN = confusion_mx[0, 0]
    FP = confusion_mx[0, 1]
    FN = confusion_mx[1, 0]
    
    print("----------")
    print("TP :", TP)
    print("TN :", TN)
    print("FP :", FP)
    print("FN :", FN)
    
    print("----------")
    
    specificity = cross_val_score(rfc, x, y, cv=20, scoring=specificity_scorer)
    print("specificity :", np.mean(specificity))
    
    recall = cross_val_score(rfc, x, y, cv=20, scoring='recall')
    print("sensitivity or recall :", np.mean(recall))

def test_rfc_complete(rfc, x, y):
    
    # Cross validation 10-Fold (for now) with accuracy scoring
    scores = cross_val_score(rfc, x, y, cv=10, scoring='accuracy')
    show_score(scores)
    
    # Prep of training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    
    # Do prediction for test values
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    
    # Let's compute the convolution matrix
    confusion_mx = metrics.confusion_matrix(y_test, y_pred)
    TP = confusion_mx[1, 1]
    TN = confusion_mx[0, 0]
    FP = confusion_mx[0, 1]
    FN = confusion_mx[1, 0]
    
    print("----------")
    print("TN :", TN, "FP :", FP, )
    print("FN :", FN, "TP :", TP, )
    
    print("----------")
    
    specificity = cross_val_score(rfc, x, y, cv=20, scoring=specificity_scorer)
    print("specificity :", np.mean(specificity))
    
    recall = cross_val_score(rfc, x, y, cv=20, scoring='recall')
    print("sensitivity or recall :", np.mean(recall))
    
    precision = cross_val_score(rfc, x, y, cv=20, scoring='precision')
    print("precision :", np.mean(precision))
    
    # getting the probability of ones of the classifier
    y_pred_prob = rfc.predict_proba(x_test)[:, 1]
    
    # separating the probability of true and false values
    y_pred_prob1 = [x[1] for x in zip(y_test, y_pred_prob) if x[0]]
    y_pred_prob0 = [x[1] for x in zip(y_test, y_pred_prob) if not x[0]]
    
    # histogram of predicted probabilities
    plt.hist(y_pred_prob1, bins=10, alpha=0.6, color="red")
    plt.hist(y_pred_prob0, bins=10, alpha=0.6, color="blue")
    plt.xlim(0, 1)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probabilities')
    plt.ylabel('Frequency')
    plt.show()
    
    # Here we do something a bit wierd, in case of a 3 valued prediction 
    # we only keep the ones who belong to 0 and 1 to make the rest of 
    # the function work
    if (len(confusion_mx) > 2):
        zipped = [(bool(x[0]), x[1]) for x in zip(y_test, y_pred_prob) if not x[0] == 2]
        # * is a way to unravel an array. With that zip unzips our previously zipped array (awesome right ;))
        y_test, y_pred_prob = zip(*zipped)
        
    # Code copied entirely from 
    # http://nbviewer.jupyter.org/github/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb
    # There wasn't really another way to show that though :)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for this classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()
    
    if (len(confusion_mx) == 2):
        # calculate cross-validated AUC score
        AUC_mean = cross_val_score(rfc, x, y, cv=10, scoring='roc_auc').mean()
        print("AUC score :", AUC_mean)
        # calculate cross-validated F1 score
        f1_mean = cross_val_score(rfc, x, y, cv=10, scoring='f1').mean()
        print("f1 score :", AUC_mean)
        
def compute_feature_importance_rfc(X, y):
    prop1 = np.sum(y) / len(y)
    prop0 = 1 - prop1
    class_weights = {
        0 : prop0,
        1 : prop1
    }
    
    feature_names = X.columns.values
    # parmaters reported from notebook
    rfc = RFC(max_features=0.8, n_estimators=33, n_jobs=-1, class_weight=class_weights)
    rfc.fit(X, y)
    return list(zip(feature_names, rfc.feature_importances_)

def show_learning_curve(rfc, x, y):

    train_sizes, train_scores, test_scores = learning_curve(rfc, x, y, cv=20, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.grid(True)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")