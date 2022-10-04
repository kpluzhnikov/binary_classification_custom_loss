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
