import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

def show_score(scores):
    print("Cross validation scores")
    print(scores)
    print("--------------------------")
    print("cross mean :", np.mean(scores))
    print("cross min :", np.min(scores))
    print("cross max :", np.max(scores))

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
    specificity = TN / float(TN + FP)
    print("specificity :", specificity)
    
    sensitivity = TP / float(TP + FN)
    print("sensitivity :", sensitivity)

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
    specificity = TN / float(TN + FP)
    print("specificity :", specificity)
    
    sensitivity = TP / float(TP + FN)
    print("sensitivity or recall :", sensitivity)
    
    precision = TP / float(TP + FP)
    print("precision :", precision)
    
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

def show_learning_curve(rfc, x, y):
    
    train_sizes, train_scores, test_scores = learning_curve(rfc, x, y, cv=20, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.grid(True)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
