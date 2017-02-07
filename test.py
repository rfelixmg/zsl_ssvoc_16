def plot_confusion_matrix(cm_, classes, normalize=True):

    import matplotlib.pyplot as plt
    import itertools

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    if normalize:
        cm_normalized = (cm_.astype('float') / cm_.sum(axis=1)[:, np.newaxis])
    else:
        cm_normalized = cm_
    cax = ax.matshow(cm_normalized,  cmap=plt.cm.gray)
    fig.colorbar(cax)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(classes[0], rotation='vertical', fontsize=22)
    ax.set_yticklabels(classes[0], rotation='horizontal', fontsize=22)

    thresh = cm_normalized.max() / 2.
    for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
        if i == j:
            plt.text(j, i, '%.2f' % (cm_normalized[i, j] * 100),
                     horizontalalignment="center",
                     fontsize=18,
                     color="green" if cm_normalized[i, j] > thresh else "red")

    plt.suptitle('Confusion Matrix (mean acc: %.2f)' % cm_normalized.diagonal().mean() * 100)
    plt.tight_layout()
    plt.show()
    fig.savefig('%s/confusion_matrix.png' % ('/tmp/'))


import numpy as np

classes = [u'persian+cat',  u'hippopotamus',  u'leopard',  u'humpback+whale',  u'seal',  u'chimpanzee',  u'rat',  u'giant+panda',  u'pig',  u'raccoon']
cm_ = np.loadtxt('/home/rfelixmg/Dropbox/PROJETOS/zsl_ssvoc_16/experiments/awa/simple_network_2_20170206_121115/confusion_matrix.txt')

plot_confusion_matrix(cm_, classes)