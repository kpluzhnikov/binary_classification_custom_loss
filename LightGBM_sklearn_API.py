import lightgbm as lgb

def custom_logregobj(beta):
  def logregobj_core(y, predt):
    '''Custom logloss function update'''
    predt = np.where(predt >= 0,
                  1. / (1. + np.exp(-predt)),
                  np.exp(predt) / (1. + np.exp(predt)))

    grad = (y + beta - beta * y) * predt - y
    hess = (y + beta - beta * y) * predt * (1 - predt)
    return grad, hess
  return logregobj_core

def custom_metric(beta):
  def logreg_metric_core(y, predt):
    '''Custom evaluation metric that should be in line with custom loss function'''
    import numpy as np
    
    predt = np.where(predt >= 0,
                 1. / (1. + np.exp(-predt)),
                 np.exp(predt) / (1. + np.exp(predt)))
    predt = np.clip(predt, 10e-7, 1-10e-7)
    loss_fn = y * np.log(predt)
    loss_fp = (1.0 - y) * np.log(1.0 - predt)
    return 'logreg_error',  np.sum(-1.0 * (loss_fn + beta * loss_fp))/len(y), False
  return logreg_metric_core
  
params = {
            'n_estimators': 500,
            'eta': 0.05,
            'disable_default_eval_metric': 1
        }   

lgb_clf = lgb.LGBMClassifier(objective=custom_logregobj(.4),**params)

lgb_clf.fit(X_train,y_train,eval_set=[(X_val,y_val)],
                eval_metric=custom_metric(.4),
                early_stopping_rounds=10,
                verbose=10)

y_pred_val = lgb_clf.predict(X_val)
y_pred_train  = lgb_clf.predict(X_train)
