import json
dtset = '/home/rfelixmg/Dropbox/PROJETOS/zsl_ssvoc_16/data/awa/'
prj = '/home/rfelixmg/Dropbox/PROJETOS/zsl_ssvoc_16/'
embd_dir = 'experiments/awa/%s/embd_test.txt' % ('simple_network_20170204_115849')

with open(dtset + 'features/tmp/ids_classname.json') as f:
    id_class = json.load(f)


def build_model():

    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    full_ids = np.loadtxt(dtset + 'features/full_ids.txt').astype(np.int) - 1
    test_ids = np.loadtxt(dtset + 'features/test_ids.txt').astype(np.int) - 1
    features = np.loadtxt(dtset + 'features/glove/features.txt').astype(np.float32)

    label_fit = np.unique(full_ids[test_ids])
    feat_fit = features[label_fit]

    clf = NearestNeighbors(n_neighbors=1, metric='euclidean')
    clf.fit(feat_fit, label_fit)
    embd_zsl = np.loadtxt(prj + embd_dir).astype(np.float32)

    return clf, full_ids[test_ids], embd_zsl

#
# def plot_roc_simple(results):
#
#     import matplotlib.pyplot as plt
#
#     plt.clf()
#     fig = plt.figure(figsize=(30, 15))
#
#     plt.plot([0, 1], [0, 1], 'k--')
#
#     keys = results.keys()
#     keys.sort()
#     accuracy = []
#     auc = []
#     for key in keys:
#         tpr_ = results[key]['tpr']
#         fpr_ = results[key]['fpr']
#         auc_ = results[key]['auc']
#         auc.append(auc_)
#
#         plt.plot(tpr_, fpr_, label='%s [auc: %.2f]' % (results[key]['label'], auc_))
#
#
#     auc = np.array(auc)
#     plt.suptitle('mean(auc): %f' % (auc.mean()))
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.title('ROC curve')
#     plt.legend(loc='best')
#     plt.show()
#     fig.savefig('roc_curve2.png')

#
# def roc_curve_prepare( y_gt, y_pred, y, y_scores, labels, id_class):
#
#     import numpy as np
#     import sklearn.metrics as metrics
#
#     results = {}
#     for label in labels:
#         ids = np.where(y_pred == label)
#         fpr, tpr, thresholds = metrics.roc_curve(y_gt == label, y_scores[ids])
#
#         area = metrics.roc_auc_score((y_gt == label), (y == label))
#
#         results[str(label)] = {
#             'label': id_class[str(label + 1)],
#             'tpr': tpr,
#             'fpr': fpr,
#             'auc': area,
#         }
#
#     return results


# def plot_roc(results):
#
#     import matplotlib.pyplot as plt
#
#     plt.clf()
#     fig = plt.figure(figsize=(30, 15))
#
#     plt.plot([0, 1], [0, 1], 'k--')
#
#     keys = results.keys()
#     keys.sort()
#     accuracy = []
#     auc = []
#     for key in keys:
#         tpr_ = results[key]['tpr']
#         fpr_ = results[key]['fpr']
#         auc_ = results[key]['auc']
#         auc.append(auc_)
#
#         plt.plot(tpr_, fpr_, label='%s [auc: %.2f]' % (results[key]['label'], auc_))
#
#     auc = np.array(auc)
#
#     plt.suptitle('mean(auc): %f' % (auc.mean()))
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.title('ROC curve')
#     plt.legend(loc='best')
#     plt.show()
#     #fig.savefig('roc_curve.png')

if __name__ == '__main__':

    import numpy as np
    from sklearn import metrics
    from utils.roc import plot_roc, get_roc_data

    clf, y_gt, embd_zsl = build_model()

    labels = np.unique(y_gt)

    y_scores, y = clf.kneighbors(embd_zsl, n_neighbors=10, return_distance=True)

    y_t = y
    rows, cols = y.shape
    for row in range(rows):
        for col in range(cols):
            y_t[row, col] = labels[y[row, col]]

    #y_scores = y_scores[:,0]
    #y_pred = []
    #for i in y[:,0]:
    #    y_pred.append(labels[i])
    #y_pred = np.array(y_pred)

    results = get_roc_data(y_gt=y_gt, y_pred=y_t, labels=labels, y_scores=y_scores)

    plot_roc(results, id_class)


