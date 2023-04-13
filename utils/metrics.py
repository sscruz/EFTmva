import numpy as np

def net_eval(bins, sm_hist, bsm_hist, n_points=100, threshold=0.5):
    
    roc = np.array([])
    
    for i in range(n_points + 1):
        thresholds = np.greater_equal(bins[:-1], i / n_points).astype(int)
        fpr, tpr = get_rates(thresholds, sm_hist, bsm_hist)
        roc = np.append(roc, [fpr, tpr])
        
    roc = roc.reshape(-1, 2)
    auc = 0
    
    for i in range(n_points):
        auc = auc + (roc[:,0][i]-roc[:,0][i+1])*(roc[:,1][i]+roc[:,1][i+1])/2
        
    bsm_cor = bsm_hist[np.greater_equal(bins[:-1], threshold).astype(bool)].sum()
    sm_cor  = sm_hist[np.less_equal(bins[1:], 1 - threshold).astype(bool)].sum()
    total   = bsm_hist.sum() + sm_hist.sum()
    a = (bsm_cor + sm_cor) / total
    
    return roc, auc, a

def get_rates(thresholds, sm_hist, bsm_hist):
    tp  = bsm_hist[np.equal(thresholds, 1)]
    fn  = bsm_hist[np.equal(thresholds, 0)]
    tn  = sm_hist[np.equal(thresholds, 0)]
    fp  = sm_hist[np.equal(thresholds, 1)]

    tpr = tp.sum() / (tp.sum() + fn.sum())
    fpr = fp.sum() / (fp.sum() + tn.sum())
    
    return fpr, tpr