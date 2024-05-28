import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize


def criterion(predictions, labels, n_classes=3):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    accuracy = accuracy_score(predictions, labels)
    f1 = f1_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')

    print(f"Accuracy: {accuracy * 100:.2f}%")
    # print(f"F1 Score: {f1:.4f}")
    # print(f"Recall: {recall:.4f}")

    # plot_multiclass_roc(predictions, labels, n_classes=n_classes)



def plot_roc_curve(predictions, labels):
    """
    绘制ROC曲线并计算AUC值。

    参数:
    - predictions: 模型预测的概率值
    - labels: 实际标签 (0 或 1)
    """
    # 计算 FPR 和 TPR
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_multiclass_roc(predictions, labels, n_classes):
    """
    绘制多分类问题的ROC曲线。

    参数:
    - predictions: 模型预测的概率值，形状为 (n_samples, n_classes)
    - labels: 实际标签，形状为 (n_samples,)
    - n_classes: 分类数量
    """
    # 将标签二值化
    labels = label_binarize(labels, classes=np.arange(n_classes))

    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制所有类别的ROC曲线
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Multi-class')
    plt.legend(loc="lower right")
    plt.show()