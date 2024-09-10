from sklearn import metrics
import numpy as np
import torch


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, config, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)   # transfer pred to [0,1]
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



#loss function and optimizer
def compile(config, model):
    # loss_func = torch.nn.BCEWithLogitsLoss()
    loss_func = BCEFocalLoss(config)
    if torch.cuda.is_available():
        loss_func = loss_func.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
    return loss_func, optimizer


def compute_metrics(all_trues, all_scores, threshold):
    all_preds = (all_scores >= threshold)

    acc = metrics.accuracy_score(all_trues, all_preds)
    pre = metrics.precision_score(all_trues, all_preds)
    rec = metrics.recall_score(all_trues, all_preds)
    f1 = metrics.f1_score(all_trues, all_preds)
    # mcc = metrics.matthews_corrcoef(all_trues, all_preds)
    fpr, tpr, _ = metrics.roc_curve(all_trues, all_scores)
    AUC = metrics.auc(fpr, tpr)
    AUPR = metrics.average_precision_score(all_trues, all_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(all_trues, all_preds, labels=[0, 1]).ravel()
    specificity = tn/(tn+fp)
    gmean = np.sqrt(rec *specificity)
    return tn, fp, fn, tp, acc, pre, rec, f1, AUC, AUPR


def print_metrics(data_type, loss, metrics):
    """ Print the evaluation results """
    tn, fp, fn, tp, acc, pre, rec, f1, AUC, AUPR= metrics
    res = '\t'.join([
        '%s:' % data_type,
        'TN=%-5d' % tn,
        'FP=%-5d' % fp,
        'FN=%-5d' % fn,
        'TP=%-5d' % tp,
        'loss:%0.5f' % loss,
        'acc:%0.4f' % acc,
        'pre:%0.4f' % pre,
        'rec:%0.4f' % rec,
        'f1:%0.4f' % f1,
        # 'mcc:%0.4f' % mcc,
        # 'gmean:%0.4f' % gmean,
        'auc:%0.4f' % AUC,
        'aupr:%0.4f' % AUPR
    ])
    print(res)

def best_acc_thr(y_true, y_score):
    """ Calculate the best threshold with acc """
    best_thr = 0.5
    best_acc = 0
    for thr in range(1,100):
        thr /= 100
        tn, fp, fn, tp, acc, pre, rec, f1, AUC, AUPR = compute_metrics(y_true, y_score, thr)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc

def best_auc_thr(y_true, y_score):
    """ Calculate the best threshold with auc """
    best_thr = 0.5
    best_auc = 0
    for thr in range(1,100):
        thr /= 100
        tn, fp, fn, tp, acc, pre, rec, f1, AUC, AUPR = compute_metrics(y_true, y_score, thr)
        if AUC>best_auc:
            best_auc = AUC
            best_thr = thr
    return best_thr, best_auc



def best_f1_thr(y_true, y_score):
    """ Calculate the best threshold with f1-score """
    best_thr = 0.5
    best_f1 = 0
    for thr in range(1, 100):
        thr /= 100
        tn, fp, fn, tp, acc, pre, rec, f1, AUC, AUPR = compute_metrics(y_true, y_score, thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1

def train_one_epoch(config, model, train_loader,loss_func, optimizer, threshold):
    model.train()

    total_train_step = 0
    total_train_loss = 0
    # start_time = time.time()

    all_trues = []
    all_scores = []
    sample_num=0
    # train model
    for idx, data in enumerate(train_loader):
        X, y = data
        sample_num += y.size(0)
        X = X.unsqueeze(0)
        # y = y.long()
        # y = y.squeeze(1)
        y = y.to(torch.float32)
        if torch.cuda.is_available():
            X = X.to(config.device)
            y = y.to(config.device)

        y_pred = model(X)
        loss = loss_func(y_pred, y)
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        # if total_train_step % 24 == 0:
        #     end_time = time.time()
        #     print("run time：", end_time - start_time)
        #     print("The loss of {}th training:{}   ".format(total_train_step, loss.item()))
        all_trues.append(y.data.cpu().numpy())
        sigmoid_y_pred = torch.sigmoid(torch.tensor(y_pred))
        all_scores.append(sigmoid_y_pred.data.cpu().numpy())
        #all_scores.append(y_pred.data.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    tn, fp, fn, tp, acc, pre, rec, f1, AUC, AUPR = compute_metrics(all_trues, all_scores, threshold)
    return all_trues, all_scores, total_train_loss / sample_num, total_train_loss, acc, pre, rec, f1, AUC, AUPR


def val_one_epoch(config, model, val_loader, loss_func, threshold):
    model.eval()

    # total_val_step = 0
    total_val_loss = 0
    all_trues = []
    all_scores = []
    sample_num = 0
    with torch.no_grad():
        for data in val_loader:
            X, y = data
            sample_num += y.size(0)
            X = X.unsqueeze(0)
            # y = y.long()
            # y = y.squeeze(1)  # torch.Size([32,])
            y = y.to(torch.float32)
            if torch.cuda.is_available():
                X = X.to(config.device)
                y = y.to(config.device)

            y_pred = model(X)
            loss = loss_func(y_pred, y)
            total_val_loss += loss.item()

            all_trues.append(y.data.cpu().numpy())
            sigmoid_y_pred = torch.sigmoid(torch.tensor(y_pred))
            all_scores.append(sigmoid_y_pred.data.cpu().numpy())
            #all_scores.append(y_pred.data.cpu().numpy())


        all_trues = np.concatenate(all_trues, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        # print('confusion matrix:\n')
        tn, fp, fn, tp, acc, pre, rec, f1, AUC, AUPR = compute_metrics(all_trues, all_scores, threshold)
        # print("TN:{}, FP:{}\n".format(tn, fp))
        # print('FN:{}, TP:{}'.format(fn, tp))
    return all_trues, all_scores, total_val_loss / sample_num, total_val_loss, acc, pre, rec, f1, AUC, AUPR

