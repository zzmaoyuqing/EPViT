import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# 5093
test_trues_MY_5093 = np.load('../../output/PPI5093/test_trues.npy')
test_scores_MY_5093 = np.load('../../output/PPI5093/final_test_scores.npy')

test_trues_MBIEP_5093 = np.load('../../output/comparision_methods_results/MBIEP/5093_test_trues.npy')
test_scores_MBIEP_5093 = np.load('../../output/comparision_methods_results/MBIEP/5093_final_test_scores.npy')

DeepEP_5093 = np.load('../../output/comparision_methods_results/DeepEP/5093_final_pred_label.npy')
test_trues_DeepEP_5093 = DeepEP_5093[1]
test_scores_DeepEP_5093 = DeepEP_5093[0]

deep_learning_5093 = np.load('../../output/comparision_methods_results/deep learning framework/5093_final_pred_label.npy')
test_trues_deep_5093 = deep_learning_5093[1]
test_scores_deep_5093 = deep_learning_5093[0]
savefig_roc_name_5093 = 'D5093 ROC curve.png'
savefig_pr_name_5093 = 'D5093 PR curve.png'

# # 3672
test_trues_MY_3672 = np.load('../../output/PPI3672/test_trues.npy')
test_scores_MY_3672 = np.load('../../output/PPI3672/final_test_scores.npy')

test_trues_MBIEP_3672 = np.load('../../output/comparision_methods_results/MBIEP/3672_test_trues.npy')
test_scores_MBIEP_3672 = np.load('../../output/comparision_methods_results/MBIEP/3672_final_test_scores.npy')

DeepEP_3672 = np.load('../../output/comparision_methods_results/DeepEP/3672_final_pred_label.npy')
test_trues_DeepEP_3672 = DeepEP_3672[1]
test_scores_DeepEP_3672 = DeepEP_3672[0]

deep_learning_3672 = np.load('../../output/comparision_methods_results/deep learning framework/3672_final_pred_label.npy')
test_trues_deep_3672 = deep_learning_3672[1]
test_scores_deep_3672 = deep_learning_3672[0]
savefig_roc_name_3672 = 'D3672 ROC curve.png'
savefig_pr_name_3672 = 'D3672 PR curve.png'


# # 2708
test_trues_MY_2708 = np.load('../../output/PPI2708/test_trues.npy')
test_scores_MY_2708 = np.load('../../output/PPI2708/final_test_scores.npy')

test_trues_MBIEP_2708 = np.load('../../output/comparision_methods_results/MBIEP/2708_test_trues.npy')
test_scores_MBIEP_2708 = np.load('../../output/comparision_methods_results/MBIEP/2708_final_test_scores.npy')

DeepEP_2708 = np.load('../../output/comparision_methods_results/DeepEP/2708_final_pred_label.npy')
test_trues_DeepEP_2708 = DeepEP_2708[1]
test_scores_DeepEP_2708 = DeepEP_2708[0]

deep_learning_2708 = np.load('../../output/comparision_methods_results/deep learning framework/2708_final_pred_label.npy')
test_trues_deep_2708 = deep_learning_2708[1]
test_scores_deep_2708 = deep_learning_2708[0]
savefig_roc_name_2708 = 'D2708 ROC curve.png'
savefig_pr_name_2708 = 'D2708 PR curve.png'


def plot_ROC_Curve(all_trues1,all_scores1,all_trues2,all_scores2,all_trues3,all_scores3,all_trues4,all_scores4,savefig_roc_name):
    # draw ROC curve
    # all_trues1,all_scores1 are EPViT    '#C52A20',
    # all_trues2,all_scores2 are MBIEP  '#96CCCB'
    # all_trues3,all_scores3 are DeepEP  '#7F7DB5'
    # all_trues4,all_scores4 are [16]  '#F3D266'

    fpr1, tpr1, _ = metrics.roc_curve(all_trues1, all_scores1)
    fpr2, tpr2, __ = metrics.roc_curve(all_trues2, all_scores2)
    fpr3, tpr3, ___ = metrics.roc_curve(all_trues3, all_scores3)
    fpr4, tpr4, ____= metrics.roc_curve(all_trues4, all_scores4)

    AUC1 = metrics.auc(fpr1, tpr1)
    AUC2 = metrics.auc(fpr2, tpr2)
    AUC3 = metrics.auc(fpr3, tpr3)
    AUC4 = metrics.auc(fpr4, tpr4)

    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(fpr1, tpr1, 'k-', color='#C52A20', label='EPViT ROC curve (area = {0:.3f})'.format(AUC1), lw=2)
    plt.plot(fpr2, tpr2, 'k--', color='#96CCCB', label='MBIEP ROC curve (area = {0:.3f})'.format(AUC2), lw=2)
    plt.plot(fpr3, tpr3, 'k-.', color='#7F7DB5', label='DeepEP ROC curve (area = {0:.3f})'.format(AUC3), lw=2)
    plt.plot(fpr4, tpr4, 'k:', color='#F3D266', label='[16] ROC curve (area = {0:.3f})'.format(AUC4), lw=2)
    plt.plot([0,1],[0,1],color='black', lw=2, linestyle='--')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(savefig_roc_name, pad_inches=0)
    # plt.show()

plot_ROC_Curve(test_trues_MY_5093, test_scores_MY_5093,test_trues_MBIEP_5093,test_scores_MBIEP_5093,test_trues_DeepEP_5093,test_scores_DeepEP_5093,test_trues_deep_5093,test_scores_deep_5093,savefig_roc_name_5093)
plot_ROC_Curve(test_trues_MY_3672, test_scores_MY_3672,test_trues_MBIEP_3672,test_scores_MBIEP_3672,test_trues_DeepEP_3672,test_scores_DeepEP_3672,test_trues_deep_3672,test_scores_deep_3672,savefig_roc_name_3672)
plot_ROC_Curve(test_trues_MY_2708, test_scores_MY_2708,test_trues_MBIEP_2708,test_scores_MBIEP_2708,test_trues_DeepEP_2708,test_scores_DeepEP_2708,test_trues_deep_2708,test_scores_deep_2708,savefig_roc_name_2708)

def plot_PR_Curve(all_trues1,all_scores1,all_trues2,all_scores2,all_trues3, all_scores3, all_trues4, all_scores4,savefig_pr_name):
    # draw PR curve
    # all_trues1,all_scores1 are EPViT     '#C52A20',
    # all_trues2,all_scores2 are MBIEP  '#96CCCB'
    # all_trues3,all_scores3 are DeepEP  '#7F7DB5'
    # all_trues4,all_scores4 are [16]  '#F3D266'

    p1, r1, _ = metrics.precision_recall_curve(all_trues1, all_scores1)
    p2, r2, __ = metrics.precision_recall_curve(all_trues2, all_scores2)
    p3, r3, ___ = metrics.precision_recall_curve(all_trues3, all_scores3)
    p4, r4, ____ = metrics.precision_recall_curve(all_trues4, all_scores4)

    AUPR1 = metrics.average_precision_score(all_trues1, all_scores1)
    AUPR2 = metrics.average_precision_score(all_trues2, all_scores2)
    AUPR3 = metrics.average_precision_score(all_trues3, all_scores3)
    AUPR4 = metrics.average_precision_score(all_trues4, all_scores4)

    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(r1, p1, 'k-', color='#C52A20', label='EPViT PR curve (area = {0:.3f})'.format(AUPR1), lw=2)
    plt.plot(r2, p2, 'k--', color='#96CCCB', label='MBIEP PR curve (area = {0:.3f})'.format(AUPR2), lw=2)
    plt.plot(r3, p3, 'k-.', color='#7F7DB5', label='DeepEP PR curve (area = {0:.3f})'.format(AUPR3), lw=2)
    plt.plot(r4, p4, 'k:', color='#F3D266', label='[16] PR curve (area = {0:.3f})'.format(AUPR4), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.savefig(savefig_pr_name, pad_inches=0)
    # plt.show()

plot_PR_Curve(test_trues_MY_5093, test_scores_MY_5093,test_trues_MBIEP_5093,test_scores_MBIEP_5093,test_trues_DeepEP_5093,test_scores_DeepEP_5093, test_trues_deep_5093, test_scores_deep_5093,savefig_pr_name_5093)
plot_PR_Curve(test_trues_MY_3672, test_scores_MY_3672,test_trues_MBIEP_3672,test_scores_MBIEP_3672,test_trues_DeepEP_3672,test_scores_DeepEP_3672,test_trues_deep_3672,test_scores_deep_3672,savefig_pr_name_3672)
plot_PR_Curve(test_trues_MY_2708, test_scores_MY_2708,test_trues_MBIEP_2708,test_scores_MBIEP_2708,test_trues_DeepEP_2708,test_scores_DeepEP_2708,test_trues_deep_2708,test_scores_deep_2708,savefig_pr_name_2708)


