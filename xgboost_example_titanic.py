# libraries / dependencies

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

import warnings

warnings.filterwarnings('ignore')

import xgboost as xgb
import sklearn
from sklearn.metrics import accuracy_score,classification_report,\
                        confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# function to draw engaging confusion matrix 
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

data=pd.read_csv('train.csv')
y = data['Survived'].copy()
X = data.drop(['Survived','PassengerId'], axis=1).copy()

# take small train size for demonstration purposes
X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, train_size=0.6,
                                                                stratify=y,random_state=0)

# Select categorical columns with relatively low cardinality
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 40 and 
                        X_train_full[cname].dtype == "object"]
# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# preprocess data without pipeline
num_imputer=SimpleImputer(strategy='mean')
X_train_num_imputed=pd.DataFrame(num_imputer.fit_transform(X_train_full[numerical_cols]))
X_val_num_imputed=pd.DataFrame(num_imputer.transform(X_val_full[numerical_cols]))

X_train_num_imputed.columns=X_train_full[numerical_cols].columns
X_val_num_imputed.columns=X_val_full[numerical_cols].columns

cat_imputer=SimpleImputer(strategy='most_frequent')

X_train_cat_imputed=pd.DataFrame(cat_imputer.fit_transform(X_train_full[categorical_cols]))
X_val_cat_imputed=pd.DataFrame(cat_imputer.transform(X_val_full[categorical_cols]))

X_train_cat_imputed.columns=X_train_full[categorical_cols].columns
X_val_cat_imputed.columns=X_val_full[categorical_cols].columns

OH_encoder=OneHotEncoder(handle_unknown='ignore',sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_cat_imputed))
OH_cols_val = pd.DataFrame(OH_encoder.transform(X_val_cat_imputed))

OH_cols_train.index = X_train_cat_imputed.index
OH_cols_val.index = X_val_cat_imputed.index

X_train=pd.concat([X_train_num_imputed,OH_cols_train],axis=1)
X_val=pd.concat([X_val_num_imputed,OH_cols_val],axis=1)

# a model without custom loss
params = {
            'n_estimators': 500,
            'eta': 0.05,
            'random_state' : 0  
        }   

xgb_classif = xgb.XGBClassifier(**params)
xgb_classif.fit(X_train,y_train,eval_set=[(X_val,y_val)],
                early_stopping_rounds=10,
                verbose=10)

y_pred_val=xgb_classif.predict(X_val)
y_pred_train=xgb_classif.predict(X_train)

print('\nValidation Accuracy : %.2f'%accuracy_score(y_val,y_pred_val))
print('Train Accuracy : %.2f'%accuracy_score(y_train,y_pred_train))

print('\nConfusion Matrix : ')
print(confusion_matrix(y_val,y_pred_val))

print('\nClassification Report : ')
print(classification_report(y_val,y_pred_val))

# the confusion matrix for the model without custom loss
plot_confusion_matrix(cm=confusion_matrix(y_val,y_pred_val),
                     target_names=['Died','Survived'],
                     normalize=False)

# custom loss user-defined functions code for sklearn version 1.0.2 (refined version)
def first_grad_logreg_beta_sklearn(y, predt):
    '''Compute the first derivative for custom logloss function'''
    return (y + beta - beta * y) * predt - y

def second_grad_logreg_beta_sklearn(y, predt):
    '''Compute the second derivative for custom logloss function'''
    return (y + beta - beta * y) * predt * (1 - predt)

def logregobj_beta_sklearn(y, predt):
    '''Custom logloss function update'''
    grad = first_grad_logreg_beta_sklearn(y, predt)
    hess = second_grad_logreg_beta_sklearn(y, predt)
    return grad, hess

def logreg_err_beta_sklearn(predt, dmat):
    '''Custom evaluation metric that should be in line with custom loss function'''
    import numpy as np
    
    y = dmat.get_label() if isinstance(dmat, xgb.DMatrix) else dmat
    predt = np.clip(predt, 10e-7, 1-10e-7)
    loss_fn = y * np.log(predt)
    loss_fp=(1.0 - y) * np.log(1.0 - predt)
    return 'logreg_error',np.sum(-1 * (loss_fn + beta * loss_fp))/len(y)

# a model with beta = .4 custom logloss to penalize FN
beta=.4

params = {
            'n_estimators': 500,
            'eta': 0.05,
            'disable_default_eval_metric': 1
        }   

xgb_classif=xgb.XGBClassifier(objective=logregobj_beta_sklearn,**params)

xgb_classif.fit(X_train,y_train,eval_set=[(X_val,y_val)],
                eval_metric=logreg_err_beta_sklearn,
                early_stopping_rounds=10,
                verbose=10)

y_pred_val=xgb_classif.predict(X_val)
y_pred_train=xgb_classif.predict(X_train)

print('\nValidation Accuracy : %.2f'%accuracy_score(y_val,y_pred_val))
print('Train Accuracy : %.2f'%accuracy_score(y_train,y_pred_train))

print('\nConfusion Matrix : ')
print(confusion_matrix(y_val,y_pred_val))

print('\nClassification Report : ')
print(classification_report(y_val,y_pred_val))

# the confusion matrix for the model with custom loss beta = .4
plot_confusion_matrix(cm=confusion_matrix(y_val,y_pred_val),
                     target_names=['Died','Survived'],
                     normalize=False)
