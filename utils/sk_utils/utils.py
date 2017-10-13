from sklearn.datasets import load_svmlight_file
from sklearn import metrics
#mem = Memory("./mycache")

#@mem.cache
def load_svm(name):
    data = load_svmlight_file(name)
    return data[0], data[1]

def auc(y,yp,pos=1,draw=False):
    fpr,tpr,thresholds = metrics.roc_curve(y,yp,pos_label=pos)
    score = metrics.auc(fpr,tpr)
    if draw:
        import matplotlib.pyplot as plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % score)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    return score

