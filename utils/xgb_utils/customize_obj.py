import numpy as np


# obj for MAE https://www.kaggle.com/wiki/MeanAbsoluteError
# J = np.mean(np.abs(y-yp))
def mae_obj(preds, dtrain):
    y  = dtrain.get_label()
    yp = preds
    c  = 2.0
    delta = yp-y
    grad  =  c*x / (np.abs(x)+c)
    hess = c**2 / (np.abs(x)+c)**2
    return rada, hess

# metric for mae
def mae_metric(preds, dtrain):
    y = dtrain.get_label()
    yp = preds
    return "MAE", np.mean(np.abs(y-yp))


# obj for logloss
def logreg_obj(preds, dtrain):
    y = dtrain.get_label()
    yp = 1.0 / (1.0 + np.exp(-preds))
    grad = yp - y
    hess = yp * (1.0-yp)
    return grad, hess

# error for logloss (discrete)
def logreg_error(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', float(sum(labels != (preds > 0.0))) / len(labels)
