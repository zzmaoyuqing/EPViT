import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

#---------------------------------ablation dataset with only PPI or only subcellular----------------------------------------------
# 5093
test_trues_MY_5093 = np.load('../../output/PPI5093/test_trues.npy')
test_scores_MY_5093 = np.load('../../output/PPI5093/final_test_scores.npy')

test_trues_onlyppi_5093 = np.load('../../output/ablation_PPI_sub/ablation_PPI_only/PPI5093/test_trues.npy')
test_scores_onlyppi_5093 = np.load('../../output/ablation_PPI_sub/ablation_PPI_only/PPI5093/final_test_scores.npy')

test_trues_onlysub_5093 = np.load('../../output/ablation_PPI_sub/ablation_sub_only/PPI5093/test_trues.npy')
test_scores_onlysub_5093 = np.load('../../output/ablation_PPI_sub/ablation_sub_only/PPI5093/final_test_scores.npy')

savefig_roc_name_5093 = 'ablation 5093 ROC curve.png'
savefig_pr_name_5093 = 'ablation 5093 PR curve.png'


# 3672
test_trues_MY_3672 = np.load('../../output/PPI3672/test_trues.npy')
test_scores_MY_3672 = np.load('../../output/PPI3672/final_test_scores.npy')

test_trues_onlyppi_3672 = np.load('../../output/ablation_PPI_sub/ablation_PPI_only/PPI3672/test_trues.npy')
test_scores_onlyppi_3672 = np.load('../../output/ablation_PPI_sub/ablation_PPI_only/PPI3672/final_test_scores.npy')

test_trues_onlysub_3672 = np.load('../../output/ablation_PPI_sub/ablation_sub_only/PPI3672/test_trues.npy')
test_scores_onlysub_3672 = np.load('../../output/ablation_PPI_sub/ablation_sub_only/PPI3672/final_test_scores.npy')

savefig_roc_name_3672 = 'ablation 3672 ROC curve.png'
savefig_pr_name_3672 = 'ablation 3672 PR curve.png'


# 2708
test_trues_MY_2708 = np.load('../../output/PPI2708/test_trues.npy')
test_scores_MY_2708 = np.load('../../output/PPI2708/final_test_scores.npy')

test_trues_onlyppi_2708 = np.load('../../output/ablation_PPI_sub/ablation_PPI_only/PPI2708/test_trues.npy')
test_scores_onlyppi_2708 = np.load('../../output/ablation_PPI_sub/ablation_PPI_only/PPI2708/final_test_scores.npy')

test_trues_onlysub_2708 = np.load('../../output/ablation_PPI_sub/ablation_sub_only/PPI2708/test_trues.npy')
test_scores_onlysub_2708 = np.load('../../output/ablation_PPI_sub/ablation_sub_only/PPI2708/final_test_scores.npy')

savefig_roc_name_2708 = 'ablation 2708 ROC curve.png'
savefig_pr_name_2708 = 'ablation 2708 PR curve.png'


def plot_ROC_Curve(all_trues1,all_scores1,all_trues2,all_scores2,all_trues3,all_scores3,savefig_roc_name):
    # draw ROC curve
    # all_trues1,all_scores1 are EPViT    '#C52A20',
    # all_trues2,all_scores2 are only PPI '#FFA500'
    # all_trues3,all_scores3 are only sub '#808080'


    fpr1, tpr1, _ = metrics.roc_curve(all_trues1, all_scores1)
    fpr2, tpr2, __ = metrics.roc_curve(all_trues2, all_scores2)
    fpr3, tpr3, ___ = metrics.roc_curve(all_trues3, all_scores3)


    AUC1 = metrics.auc(fpr1, tpr1)
    AUC2 = metrics.auc(fpr2, tpr2)
    AUC3 = metrics.auc(fpr3, tpr3)


    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(fpr1, tpr1, 'k-', color='#C52A20', label='PPI+sub ROC curve (area = {0:.3f})'.format(AUC1), lw=2)
    plt.plot(fpr2, tpr2, 'k--', color='#FFA500', label='only PPI ROC curve (area = {0:.3f})'.format(AUC2), lw=2)
    plt.plot(fpr3, tpr3, 'k-.', color='#808080', label='only sub ROC curve (area = {0:.3f})'.format(AUC3), lw=2)
    plt.plot([0,1],[0,1],color='black', lw=2, linestyle='--')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(savefig_roc_name, pad_inches=0)
    # plt.show()

plot_ROC_Curve(test_trues_MY_5093, test_scores_MY_5093,test_trues_onlyppi_5093,test_scores_onlyppi_5093,test_trues_onlysub_5093,test_scores_onlysub_5093,savefig_roc_name_5093)
plot_ROC_Curve(test_trues_MY_3672, test_scores_MY_3672,test_trues_onlyppi_3672,test_scores_onlyppi_3672,test_trues_onlysub_3672,test_scores_onlysub_3672,savefig_roc_name_3672)
plot_ROC_Curve(test_trues_MY_2708, test_scores_MY_2708,test_trues_onlyppi_2708,test_scores_onlyppi_2708,test_trues_onlysub_2708,test_scores_onlysub_2708,savefig_roc_name_2708)

def plot_PR_Curve(all_trues1,all_scores1,all_trues2,all_scores2,all_trues3, all_scores3,savefig_pr_name):
    # draw PR curve
    # all_trues1,all_scores1 are EPViT     '#C52A20',
    # all_trues2,all_scores2 are only PPI  '#FFA500'
    # all_trues3,all_scores3 are only sub  '#808080'

    p1, r1, _ = metrics.precision_recall_curve(all_trues1, all_scores1)
    p2, r2, __ = metrics.precision_recall_curve(all_trues2, all_scores2)
    p3, r3, ___ = metrics.precision_recall_curve(all_trues3, all_scores3)


    AUPR1 = metrics.average_precision_score(all_trues1, all_scores1)
    AUPR2 = metrics.average_precision_score(all_trues2, all_scores2)
    AUPR3 = metrics.average_precision_score(all_trues3, all_scores3)


    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(r1, p1, 'k-', color='#C52A20', label='PPI+sub PR curve (area = {0:.3f})'.format(AUPR1), lw=2)
    plt.plot(r2, p2, 'k--', color='#FFA500', label='only PPI PR curve (area = {0:.3f})'.format(AUPR2), lw=2)
    plt.plot(r3, p3, 'k-.', color='#808080', label='only sub PR curve (area = {0:.3f})'.format(AUPR3), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.savefig(savefig_pr_name, pad_inches=0)
    # plt.show()

plot_PR_Curve(test_trues_MY_5093, test_scores_MY_5093,test_trues_onlyppi_5093,test_scores_onlyppi_5093,test_trues_onlysub_5093,test_scores_onlysub_5093,savefig_pr_name_5093)
plot_PR_Curve(test_trues_MY_3672, test_scores_MY_3672,test_trues_onlyppi_3672,test_scores_onlyppi_3672,test_trues_onlysub_3672,test_scores_onlysub_3672,savefig_pr_name_3672)
plot_PR_Curve(test_trues_MY_2708, test_scores_MY_2708,test_trues_onlyppi_2708,test_scores_onlyppi_2708,test_trues_onlysub_2708,test_scores_onlysub_2708,savefig_pr_name_2708)

