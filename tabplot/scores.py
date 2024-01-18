import numpy as np

def sse(y0, y):
    """
    Sum of squares of errors
    """
    return np.sum((y0 - y)**2, axis=0)

def rmse(y0, y): 
    return np.sqrt(np.average((y0 - y)**2, axis=0))

def nrmse(y0, y): 
    return np.sqrt(np.average((y0 - y)**2, axis=0)) / (np.max(y0, axis=0) - np.min(y0, axis=0)) 

def logsse(y0, y): 
    return np.log10(np.sum((y0 - y)**2, axis=0))

def logrmse(y0, y):
    return np.log10(np.sqrt(np.average((y0 - y)**2, axis=0)))

scores_dict = {
    'sse': sse,
    'rmse': rmse,
    'nrmse': nrmse,
    'logrmse': logrmse,
    'logsse': logsse
}

def evaluate_scores(y0, y, score_type): 
    return scores_dict[score_type](y0, y)

