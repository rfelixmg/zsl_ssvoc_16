#!/usr/bin/env python

"""

sample:
python train.py -exp 'test001' >> ./experiments/awa/test_001.txt


"""
from sklearn.preprocessing import label

__author__ = "Rafael Felix"
__license__ = "UoA"
__copyright__ = "Copyright 2016"
__maintainer__ = "Rafael Felix"
__email__ = "rfelixmg@gmail.com"
__status__ = "beta"

import random, os, sys, argparse

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

import json, datetime, time, h5py
import numpy as np

from utils.roc import roc
from utils import file_utils

sys.path.append(os.path.abspath('..'))

tag_ = 'awa'
dataset_ = './data/%s/' % tag_

configuration = {
    'dataset': dataset_,
    'dataset_image': dataset_ + 'images/',
    'dataset_text': dataset_ + 'fine_grained_description/',
    'dataset_attributes': dataset_ + 'attributes/',
    'embedding': dataset_ + 'features/',
    # 'embedding_image':          dataset_ + 'features/halah_googlenet/feature.txt',
    'embedding_image': dataset_ + 'features/lampert_vgg/feature.h5',
    'embedding_text': dataset_ + 'features/bow_text/None',
    #'embedding_attributes':     dataset_ + 'features/bow_attributes/feature.txt',
    'embedding_attributes': dataset_ + 'features/word2vec/feature.txt',
    # 'estimation_attributes': dataset_ + '/attributes/class_attribute_labels_continuous.txt',
    #   mlp,
    'lr': 0.01,
    'lamda_regularizer': 0.000001,
    '#classes': None,
    'baseline_model': 'mlp',
    '#neighbors': 1,
    'number_epochs': 10,
    'n_estimators': 100,
    'max_iter': 200,
    'n_jobs': -2,
    'estimated_values': False,
    'output_file': '',
    'tag': tag_,
    'C': 10.,
    'experiment_name': 'experiment001'
}

if configuration['tag'] == 'cub':
    configuration['#classes'] = 200
elif configuration['tag'] == 'awa':
    configuration['#classes'] = 50

evaluation = {}


def visual_mode():
    try:

        from keras import callbacks
        remote = callbacks.RemoteMonitor(root='http://localhost:9000')

        import subprocess
        subprocess.Popen("python ./modules/hualos/api.py 1", shell=True)
    except:
        print 'Error: API hualos'

    import webbrowser
    webbrowser.get('firefox').open_new('http://localhost:9000')

    return remote


def plot_roc(matrix_results, id_class):
    plt.clf()
    fig = plt.figure(figsize=(30, 15))
    plt.plot([0, 1], [0, 1], 'k--')

    keys = matrix_results.keys()
    keys.sort()
    for key in keys:
        tpr_ = matrix_results[key]['tpr']
        fpr_ = matrix_results[key]['fpr']
        auc_ = matrix_results[key]['auc']

        plt.plot(fpr_, tpr_, label='%s (%d)[auc: %.2f]' % (id_class[str(key + 1)], key + 1, auc_))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    fig.savefig('%s/roc_curve.png' % (configuration['exp_folder']))


def collect_splits(directory):
    id_labels = np.loadtxt(directory + 'id_labels.txt')[:, 1].astype(np.int) - 1
    labels = np.unique(id_labels)

    try:
        id_train_samples = np.loadtxt(directory + 'train_ids.txt').astype(np.int) - 1
        id_test_samples = np.loadtxt(directory + 'test_ids.txt').astype(np.int) - 1
    except:

        # Load dataset splits
        set_train_classes = np.loadtxt(directory + 'train_classes.txt').astype(np.int) - 1
        set_test_classes = np.loadtxt(directory + 'test_classes.txt').astype(np.int) - 1

        id_train_samples = np.array((), dtype=np.int)
        id_test_samples = np.array((), dtype=np.int)

        for label in set_train_classes:
            all_pos = np.where(id_labels == label)[0]
            id_train_samples = np.concatenate((id_train_samples, all_pos))

        for label in set_test_classes:
            all_pos = np.where(id_labels == label)[0]
            id_test_samples = np.concatenate((id_test_samples, all_pos))

        id_train_samples = np.random.permutation(id_train_samples)

        np.savetxt(directory + 'train_ids.txt', id_train_samples + 1, fmt='%d')
        np.savetxt(directory + 'test_ids.txt', id_test_samples + 1, fmt='%d')

    return labels, id_labels, id_train_samples, id_test_samples


def load_args():
    # Getting arg parameters
    parser = argparse.ArgumentParser(description='Training LSTM to generate text in char-rnn level')
    parser.add_argument('-d', '-dataset',
                        default=str(configuration['dataset']),
                        help='Textual dataset folder', required=False)

    parser.add_argument('-emb_dim', '-embedding_dimension',
                        help='Dimension of embedding for textual and visual representation',
                        default=int(1024),
                        required=False)

    parser.add_argument('-att', '-attributes',
                        help='Attributes dataset file',
                        default=configuration['embedding_attributes'],
                        required=False)

    parser.add_argument('-cnn', '-cnn_features',
                        help='Features in dataset file',
                        default=str(configuration['embedding_image']),
                        required=False)

    parser.add_argument('-tenc', '-text_encoder',
                        default=str('bow'),
                        help='Textual Encoder: "bow", "lstm"', required=False)

    parser.add_argument('-exp', '-experiment_name', default=configuration['experiment_name'],
                        help='experiment name', required=False)

    parser.add_argument('-lr', '-learning_rate', default=configuration['lr'],
                        help='learning rate', required=False)

    parser.add_argument('-verbose', default=False,
                        help='Verbose debug mode', required=False)

    return vars(parser.parse_args())


def euclidean_distance(y_true, y_pred):
    from keras.optimizers import K
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=1))


def distance(y_true, y_pred):
    return np.sqrt(np.sum(np.square(y_true - y_pred), axis=1))


def save_model(clf):
    file_config = configuration['exp_folder'] + 'model.json'
    file_weight = configuration['exp_folder'] + 'model_weights.h5'
    model_json = clf.to_json()
    with open(file_config, "w") as json_file:
        json_file.write(model_json)
    clf.save_weights(file_weight)
    print("Saved model to disk")


def build_model():
    from keras.layers import Input, Dense, Activation
    from keras.models import Model
    from keras.regularizers import l2
    from keras.optimizers import SGD

    input_x = Input(shape=(4096,), name='input_x')

    f_x = Dense(300,
                W_regularizer=l2(configuration['lamda_regularizer']),
                name='W_embedding',
                activation='linear')(input_x)

    # input_u = Input(shape=(300,), name='input_u')
    # g_u = Dense(300, activation='relu', init='identity', name='V_embedding')(input_u)

    sgd = SGD(lr=configuration['lr'], decay=1e-10, momentum=.9)
    model = Model(input=input_x, output=f_x)

    model.compile(optimizer=sgd,
                  loss=euclidean_distance)
    # loss='mean_squared_error')

    return model


def main(args):
    print '#' * 50, '\n'
    print 'Configuration: \n', json.dumps(configuration, sort_keys=True, indent=4)
    print '#' * 50, '\n'
    print '-' * 50, '\nLoading data ...\n', '-' * 50
    print '\n', '#' * 50

    with open(configuration['exp_folder'] + 'exp_setup.json', 'w') as outfile:
        obj_ = {'configuration': configuration}
        json.dump(obj_, outfile, sort_keys=True, indent=4)

    # Load attributes
    attributes_data = np.loadtxt(configuration['embedding_attributes'])
    # Normalize [-1, +1]
    # attributes_data = (attributes_data * 2) - 1
    print 'Attributes shape: ', attributes_data.shape

    # Load googlenet
    start = time.time()
    # cnn_data = np.loadtxt(cnn_dataset_file)
    cnn_data = np.array(h5py.File(configuration['embedding_image'], 'r')['vgg'])
    cnn_data = np.ascontiguousarray(cnn_data)
    print 'CNN shape: ', cnn_data.shape
    print 'Time (load dataset): %f' % (time.time() - start)

    print '-' * 50, '\nData loaded ...\n', '-' * 50

    labels, id_labels, train_set, test_set = collect_splits(configuration['dataset'] + 'features/')

    with open('./data/awa/features/tmp/ids_classname.json') as outp:
        id_class = json.load(outp)
    with open('./data/awa/features/tmp/predicates.json') as outp:
        predicates = json.load(outp)

    X_train = cnn_data[train_set]  # [:1000]
    A_train = attributes_data[train_set]  # [:1000]
    Y_train = id_labels[train_set]  # [:1000]

    X_test = cnn_data[test_set]
    A_test = attributes_data[test_set]
    Y_test = id_labels[test_set]

    labels_test = np.unique(Y_test)

    # X_train = np.ones((100, 4096))
    # A_train = np.ones((100, 300))
    # Y_train = np.ones(100)
    #
    # for i in range(10):
    #     X_train[(10*(i)): ((10*(i+1)))] *= (0.1 * i)
    #     A_train[(10*(i)): ((10*(i+1)))] += i
    #     Y_train[(10*(i)): ((10*(i+1)))] = i
    #
    # X_test = X_train
    # A_test = A_train
    # Y_test = Y_train


    del cnn_data

    # Embedding model
    print '-' * 50, '\nTraining embedding model\n', '-' * 50
    clf = build_model()
    start = time.time()

    remote = visual_mode()

    from keras.callbacks import ModelCheckpoint

    filepath = configuration['exp_folder'] + 'weights/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

    ######################################################################
    ######################################################################
    history = clf.fit(X_train, A_train,
                      # validation_data=(X_train, A_train),
                      validation_split=0.1,
                      shuffle=False,
                      batch_size=32,
                      nb_epoch=configuration['number_epochs'],
                      callbacks=[remote, checkpoint])

    with open(configuration['exp_folder'] + 'history.json', 'w') as outfile:
        obj_ = {'history': history.history}
        json.dump(obj_, outfile, sort_keys=True, indent=4)
    ######################################################################
    ######################################################################

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show(block=False)
    plt.savefig('%s/loss_val_loss.png' % (configuration['exp_folder']))

    print 'Time (load training): %f' % (time.time() - start)
    print '-' * 50, '\nSaving embedding model\n', '-' * 50
    save_model(clf)

    print '-' * 50, '\nEmbedding model Trained\n', '-' * 50

    evaluation['score_train'] = clf.evaluate(X_train, A_train)
    evaluation['score_test'] = clf.evaluate(X_test, A_test)

    embd_test = clf.predict(X_test)

    np.savetxt(configuration['exp_folder'] + 'embd_test.txt', embd_test, fmt='%f')

    # Distance model
    print '-' * 50, '\nTraining distance model\n', '-' * 50
    knn = KNeighborsClassifier(n_neighbors=configuration['#neighbors'], metric='euclidean')
    print '\n\n>> Config K-NN with attributes data'
    knn.fit(A_test, Y_test)
    print '-' * 50, '\nDistance model prepared\n', '-' * 50

    # Apply distance model
    print '-' * 50, '\nRunning Zero-Shot Learning\n', '-' * 50
    zsl_test = knn.predict(A_test)

    print '-' * 50, '\nRunning Evaluation\n', '-' * 50
    evaluation['accuracy_test'] = metrics.accuracy_score(Y_test,
                                                         zsl_test)
    evaluation['precision_test'] = metrics.precision_score(Y_test,
                                                           zsl_test,
                                                           labels=labels,
                                                           average='weighted')
    evaluation['recall_test'] = metrics.recall_score(Y_test,
                                                     zsl_test,
                                                     labels=labels,
                                                     average='weighted')

    cm_ = metrics.confusion_matrix(Y_test,
                                   zsl_test,
                                   labels=labels_test)
    np.savetxt(configuration['exp_folder'] + 'confusion_matrix.txt', cm_, fmt='%d')

    test_classes = []
    test_classes.append([str(id_class[str(ic + 1)]) for ic in labels_test])
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_)
    fig.colorbar(cax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(test_classes[0], rotation='vertical')
    ax.set_yticklabels(test_classes[0], rotation='horizontal')
    plt.gray()
    fig.savefig('%s/confusion_matrix.png' % (configuration['exp_folder']))

    print '-' * 50, '\nRunning Evaluation per Class\n', '-' * 50

    eval_per_class = {}
    matrix_results = {}
    for c in labels_test:
        y = (Y_test == c) * 1
        y_ = (zsl_test == c) * 1
        acc_ = metrics.accuracy_score(Y_test == c,
                                      zsl_test == c)

        pr_ = metrics.precision_score(Y_test == c,
                                      zsl_test == c)

        re_ = metrics.recall_score(Y_test == c,
                                   zsl_test == c)

        cm_ = metrics.confusion_matrix(Y_test == c,
                                       zsl_test == c)
        # roc from Lampert
        tpr_, fpr_, auc_ = roc(None, y_, y)

        eval_per_class[c] = {'accuracy': acc_,
                             'precision': pr_,
                             'recall': re_
                             }
        matrix_results[c] = {'confusion_matrix': cm_,
                             'fpr': fpr_,
                             'tpr': tpr_,
                             'auc': auc_}

    evaluation['~evaluation_per_class'] = eval_per_class

    print json.dumps(evaluation, sort_keys=True, indent=4)

    with open(configuration['exp_folder'] + 'evaluation.json', 'w') as outfile:
        obj_ = {'evaluation': evaluation}
        json.dump(obj_, outfile, sort_keys=True, indent=4)

    # plot roc
    plot_roc(matrix_results, id_class)

    return evaluation, clf


if __name__ == '__main__':
    print '\n\n\nInitializing application...\n\n'

    random.seed(0)
    args = load_args()
    # print args

    configuration['lr'] = float(args['lr'])
    configuration['experiment_name'] = args['exp'] + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    configuration['exp_folder'] = './experiments/%s/%s/' % (configuration['tag'], configuration['experiment_name'])

    configuration['output_file'] = '%s%s_evaluation_%dnn%s.txt' % \
                                   (configuration['exp_folder'],
                                    configuration['baseline_model'],
                                    configuration['#neighbors'],
                                    '_estimated_values' if configuration['estimated_values'] else '')

    file_utils.makedir(configuration['experiment_name'], './experiments/%s/' % configuration['tag'])
    file_utils.makedir('weights', configuration['exp_folder'])

    t_global = time.time()

    #evaluation, clf = main(args)
    print 'Total (time): %f' % (time.time() - t_global)

    print '---\nClosing application ...\n'
