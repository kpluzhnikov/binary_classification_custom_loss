# Libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score,classification_report,\
                        confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


sklearn.__version__ # should be 1.0.2 to sklearn custom loss work properly

# Casual Titanic dataset preprocessing
data=pd.read_csv('train.csv')
y = data['Survived'].copy()
X = data.drop(['Survived','PassengerId'], axis=1).copy()

X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, train_size=0.6,
                                                                stratify=y,random_state=0)

# Select categorical columns with relatively low cardinality
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 40 and 
                        X_train_full[cname].dtype == "object"]
# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

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

########### XGBoost core API ##########
def first_grad_logreg_beta(predt, dtrain):
    '''Compute the first derivative for custom logloss function'''
    y = dtrain.get_label() if isinstance(dtrain, xgb.DMatrix) else dtrain
    return (y + beta - beta * y) * predt - y

def second_grad_logreg_beta(predt, dtrain):
    '''Compute the second derivative for custom logloss function'''
    y = dtrain.get_label() if isinstance(dtrain, xgb.DMatrix) else dtrain
    return (y + beta - beta * y) * predt * (1 - predt)

def logregobj_beta(predt, dtrain):
    '''Custom logloss function update'''
    grad = first_grad_logreg_beta(predt, dtrain)
    hess = second_grad_logreg_beta(predt, dtrain)
    return grad, hess

def logreg_err_beta(predt, dmat):
    '''Custom evaluation metric that should be in line with custom loss function'''
    import numpy as np
    
    y = dmat.get_label() if isinstance(dmat, xgb.DMatrix) else dmat
    predt=np.clip(predt, 10e-7, 1-10e-7)
    loss_fn=y * np.log(predt)
    loss_fp=(1.0 - y) * np.log(1.0 - predt)
    return 'logreg_error', np.sum(-1 * (loss_fn + beta * loss_fp))/len(y)
  
beta=.4

params = {
            'eta': 0.05,
            'disable_default_eval_metric': 1
        }   

X_train.columns=[str(col) for col in X_train.columns]
X_val.columns=X_train.columns

dmat_train=xgb.DMatrix(X_train,y_train,feature_names=X_train.columns)
dmat_val=xgb.DMatrix(X_val,y_val,feature_names=X_val.columns)

xgb_model=xgb.train(params,
                   dmat_train,
                   obj=logregobj_beta,
                   evals=[(dmat_train,'train'),(dmat_val,'validation')],
                   num_boost_round=500,
                   early_stopping_rounds=10,
                   feval=logreg_err_beta,
                    )

y_pred_train=[1 if pred>0.5 else 0 for pred in xgb_model.predict(data=dmat_train)]
y_pred_val=[1 if pred>0.5 else 0 for pred in xgb_model.predict(data=dmat_val)]

############ XGBoost sklearn API (sklearn v.1.0.2) ##########

# code for sklearn version 1.0.2 (refined version)
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

y_hat = xgb_classif.predict_proba(X_val)
