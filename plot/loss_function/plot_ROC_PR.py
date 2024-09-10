import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
#

#---------------------------------focal loss VS BCEloss----------------------------------------------
# # 5093
# test_trues1 = np.load('../../output/PPI5093/test_trues.npy')
# test_scores1 = np.load('../../output/PPI5093/final_test_scores.npy')
#
# test_trues2 = np.load('../../output/ablation_BCELOSS/PPI5093/test_trues.npy')
# test_scores2 = np.load('../../output/ablation_BCELOSS/PPI5093/final_test_scores.npy')
# savefig_roc_name = 'loss_5093 ROC curve.png'
# savefig_pr_name = 'loss_5093 PR curve.png'

# # 3672
# test_trues1 = np.load('../../output/PPI3672/test_trues.npy')
# test_scores1 = np.load('../../output/PPI3672/final_test_scores.npy')
#
# test_trues2 = np.load('../../output/ablation_BCELOSS/PPI3672/test_trues.npy')
# test_scores2 = np.load('../../output/ablation_BCELOSS/PPI3672/final_test_scores.npy')
# savefig_roc_name = 'loss_3672 ROC curve.png'
# savefig_pr_name = 'loss_3672 PR curve.png'


# 2708
test_trues1 = np.load('../../output/PPI2708/test_trues.npy')
test_scores1 = np.load('../../output/PPI2708/final_test_scores.npy')

test_trues2 = np.load('../../output/ablation_BCELOSS/PPI2708/test_trues.npy')
test_scores2 = np.load('../../output/ablation_BCELOSS/PPI2708/final_test_scores.npy')
savefig_roc_name = 'loss_2708 ROC curve.png'
savefig_pr_name = 'loss_2708 PR curve.png'

def plot_loss_ROC_Curve(all_trues1,all_scores1,all_trues2,all_scores2, savefig_roc_name):
    # draw ROC curve
    # all_trues1,all_scores1 are focal loss     '#C52A20'
    # all_trues2,all_scores2 are BCEWithLogitsLoss  '#669BBB'


    fpr1, tpr1, _ = metrics.roc_curve(all_trues1, all_scores1)
    fpr2, tpr2, __ = metrics.roc_curve(all_trues2, all_scores2)


    AUC1 = metrics.auc(fpr1, tpr1)
    AUC2 = metrics.auc(fpr2, tpr2)


    plt.figure(figsize=(6,6), dpi=300)
    plt.plot(fpr1, tpr1, 'k-', color='#C52A20', label='FL ROC (area = {0:.3f})'.format(AUC1), lw=2)
    plt.plot(fpr2, tpr2, 'k--', color='#669BBB', label='BCE ROC (area = {0:.3f})'.format(AUC2), lw=2)
    plt.plot([0,1],[0,1],color='black', lw=2, linestyle='--')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(savefig_roc_name, pad_inches=0)
    # plt.show()

plot_loss_ROC_Curve(test_trues1, test_scores1,test_trues2,test_scores2,savefig_roc_name)



def plot_loss_PR_Curve(all_trues1,all_scores1,all_trues2,all_scores2,savefig_pr_name):
    # draw PR curve
    # all_trues1,all_scores1 are focal loss     '#C52A20'
    # all_trues2,all_scores2 are BCEWithLogitsLoss  '#669BBB'

    p1, r1, _ = metrics.precision_recall_curve(all_trues1, all_scores1)
    p2, r2, __ = metrics.precision_recall_curve(all_trues2, all_scores2)


    AUPR1 = metrics.average_precision_score(all_trues1, all_scores1)
    AUPR2 = metrics.average_precision_score(all_trues2, all_scores2)

    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(r1, p1, 'k-', color='#C52A20', label='FL PR curve (area = {0:.3f})'.format(AUPR1), lw=2)
    plt.plot(r2, p2, 'k--', color='#fbf5d0', label='BCE PR curve (area = {0:.3f})'.format(AUPR2), lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.savefig(savefig_pr_name, pad_inches=0)
    # plt.show()

plot_loss_PR_Curve(test_trues1, test_scores1,test_trues2,test_scores2, savefig_pr_name)

