import numpy as np
from sklearn import metrics

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

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
    
    