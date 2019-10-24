import numpy as np
from sklearn.metrics import r2_score,log_loss,mean_squared_log_error
import scipy


def r2_coef(pred,train):
    return "r2", -r2_score(train.get_label(),pred)

def RMSLE(pred,train):
    target = train.get_label()
    return "RMSLE", mean_squared_log_error(target, pred), False


def KL_loss(pred,train):
    def get_KLdive(μ1,σ1,μ2,σ2):
        A = np.log(σ2/σ1)
        B = ( σ1**2 + (μ1 - μ2)**2 ) / (2*σ2**2)
        C = -1/2
        y = A + B + C
        return y
    
    mu1 = pred.mean()
    sigma1 = pred.std()
    true = train.get_label()
    mu2 = true.mean()
    sigma2 = true.std()
    kl_loss = get_KLdive(mu1,sigma1,mu2,sigma2)

    low,high = 1.,2.7
    neg_idx = np.where(true < low)[0]
    pos_idx = np.where(true > high)[0]
    loss = abs(true- pred)
    neg_loss = sum(loss[neg_idx])/len(neg_idx)
    pos_loss = sum(loss[pos_idx])/len(pos_idx)
    loss = kl_loss +  np.sqrt(neg_loss + pos_loss)
    return "KL-Loss", loss, False

def outlier_loss(pred,train):
    score = train.get_label()
    l = len(pred)
    neg_idx = np.where(score < 0.4)[0]
    #pos_idx = np.where(score > 4)[0]
    idx = np.where((0.4<=score)& (score <= 4))[0]
    loss = abs(score - pred)/l
    neg_r2_loss = 1 - r2_score(pred[neg_idx],score[neg_idx])
    #pos_r2_loss = 1-r2_score(pred[pos_idx],score[pos_idx])
    #r2_loss = 1-r2_score(pred,score)
    #w1,w2,w3 = (l-len(idx))/l,(l-len(neg_idx))/l,(l-len(pos_idx))/l
    w1,w2 = 1/len(idx),1/len(neg_idx)
    loss = sum(loss[idx])*w1 + (neg_r2_loss + sum(loss[neg_idx]))*w2# + (pos_r2_loss+ sum(loss[pos_idx]))*w3
    #loss = np.sqrt(r2_loss*w1 + neg_r2_loss*w2)#+ pos_r2_loss*w3)
    #tmp = (sum(loss[neg_idx])/sum(loss)) * 10
    #loss = sum(loss[idx])*w1 + (sum(loss[neg_idx]))*w2
    loss = np.sqrt(loss)
    return "OLLoss",loss, False

def CB_softmax_entorpy(pred,train):
    def E(n_y):
        beta = 0.99
        return (1 - beta**n_y)/(1-beta)
    low,high = 0.4,3.8
    score = train.get_label()
    l = len(pred)
    neg_idx = np.where(score < low)[0]
    pos_idx = np.where(score > high)[0]
    idx = np.where((low<=score)&(high>=score))[0]
    idx = np.where(low<=score[0])
    neg_r2_loss = 1 - r2_score(score[neg_idx],pred[neg_idx])
    r2_loss = 1-r2_score(score,pred)
    pos_r2_loss = 1-r2_score(score[pos_idx],pred[pos_idx])
    loss1 = neg_r2_loss/E(len(neg_idx))
    loss2 = r2_loss/E(len(idx))
    loss3 = pos_r2_loss/E(len(pos_idx))
    loss = loss1 + loss2 + loss3
    
    return "SELoss",loss,False

def CB_facal_loss(pred,train):
    def E(n_y):
        beta = 0.99
        return (1 - beta**n_y)/(1-beta)
    low,high = 0,4
    gamma = 1.
    score = train.get_label()
    l = len(pred)
    neg_idx = np.where(score < low)[0]
    pos_idx = np.where(score > high)[0]
    idx = np.where((low<=score)&(high>=score))[0]
    pred_ = np.where((low<=pred)&(high>=pred),1,0)
    
    neg_r2_loss = 1 - r2_score(pred[neg_idx],score[neg_idx])
    r2_loss = 1-r2_score(pred,score)
    pos_r2_loss = 1-r2_score(pred[pos_idx],score[pos_idx]) 

    loss1 = (1 - sum(pred < low)/len(neg_idx))* np.sqrt(neg_r2_loss)*2
    loss2 = (1 - sum(pred > high)/len(pos_idx))* np.sqrt(pos_r2_loss)*2
    loss3 = (1 - sum(pred_)/len(idx)) * np.sqrt(r2_loss) * 2
    loss = loss1/E(len(neg_idx))+loss2/E(len(pos_idx))+ loss3/E(len(idx))
    return "FLLoss",loss,False


def KU_loss(pred, train):
    def E(n_y):
        beta = 0.99
        return (1 - beta**n_y)/(1-beta)

    score = train.get_label()
    r2_loss = 1-r2_score(score,pred)
    ku_pred = scipy.stats.kurtosis(pred)
    ku_tgt = scipy.stats.kurtosis(score)
    KU_loss = (ku_tgt - ku_pred) ** 4
    
    loss = r2_loss + KU_loss
    return "KU_Loss", loss, False
