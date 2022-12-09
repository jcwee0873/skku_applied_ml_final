from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, NearMiss

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

CLASS_LABEL = {
    0: "Angry",
    1: "Happy",
    2: "Sad",
    3: "Neutral"
}


def over_sampling(X_train, y_train, o_sampling=None, u_sampling=None, **kargs):
    print(f"Before: X({X_train.shape}), y({y_train.shape})")
    print(f"Before Class: {Counter(y_train)}")
    if o_sampling == 'adasyn':
        smote = ADASYN(**kargs)
    else:
        smote = SMOTE(**kargs)
        
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"After Over Sampling: X({X_train.shape}), y({y_train.shape})")
    print(f"After Over Sampling Class: {Counter(y_train)}")

    if u_sampling:
        if u_sampling == 'nearmiss':
            under = NearMiss()
        else:
            under = TomekLinks()
        X_train, y_train = under.fit_resample(X_train, y_train)
        
        print(f"After UnderSampling: X({X_train.shape}), y({y_train.shape})")
        print(f"After UnderSampling Class: {Counter(y_train)}")


    return X_train, y_train

def preprocessing(
    X_data, y_data,
    test_ratio=.3,
    random_state=202,
    normalize=255,
    o_sampling=None,
    u_sampling=None,
    **kargs
):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_ratio, random_state=random_state, stratify=y_data)

    if o_sampling:
        X_train, y_train = over_sampling(X_train, y_train, o_sampling=o_sampling, u_sampling=u_sampling, **kargs)

    X_train = X_train.reshape(-1, 36, 36) / normalize
    X_test = X_test.reshape(-1, 36, 36) / normalize
#     y_train = torch.tensor(y_train)
#     y_test = torch.tensor(y_test)

    X = [X_train, X_test]
    y = [y_train, y_test]

    return X, y


def evaluate(model, X, y, is_torch=True, label_map=CLASS_LABEL):
    if is_torch:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(device)
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_hat, z = model(X)
            if isinstance(y_hat, tuple):
                y_hat = y_hat[0]

        y_pred = np.array(y_hat.cpu().argmax(dim=-1))
        y = np.array(y.cpu())
        
    else:
        y_pred = model.predict(X)

    print('\nClassification Report')
    print(classification_report(y_pred, y))

    cm = confusion_matrix(y, y_pred, normalize='true')
    cm2 = confusion_matrix(y, y_pred)

    label_index = list(label_map.keys())
    label_name = list(label_map.values())
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap='Blues')
    for i in label_index:
        for j in label_index:
            plt.text(
                i, j, '{}\n({}%)'.format(cm2[j][i], round(cm[j][i] * 100, 1)) ,
                horizontalalignment="center",
                verticalalignment="center",
                color='black' if cm[j][i] < 0.5 else 'white'
                
            )
    plt.xticks(label_index, label_name)
    plt.yticks(label_index, label_name)
    plt.xlabel('y')
    plt.ylabel('y_pred', rotation=0)
    plt.title('Confusion Matrix')
    plt.show()