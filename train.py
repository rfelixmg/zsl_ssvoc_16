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

import random, os, sys, argparse, itertools
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
    # 'embedding_attributes':     dataset_ + 'features/bow_attributes/feature.txt',
    # 'embedding_attributes': dataset_ + 'features/word2vec/feature.txt',
    'embedding_attributes': dataset_ + 'features/glove/feature.txt',
    # 'estimation_attributes': dataset_ + '/attributes/class_attribute_labels_continuous.txt',
    #   mlp,
    'lr': 0.01,
    'lr_decay': 1e-3,
    'lamda_regularizer': 0.0000001,
    '#classes': None,
    'baseline_model': 'mlp',
    '#neighbors': 1,
    'number_epochs': 100,
    'n_estimators': 100,
    'max_iter': 200,
    'n_jobs': -2,
    'estimated_values': False,
    'output_file': '',
    'tag': tag_,
    'C': 10.,
    'experiment_name': 'test',
    'debug': True
}

if configuration['tag'] == 'cub':
    configuration['#classes'] = 200
elif configuration['tag'] == 'awa':
    configuration['#classes'] = 50

evaluation = {}

from keras.callbacks import Callback
import warnings


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period == 0:
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


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
            plt.text(j, i, '%.2f' % (cm_normalized[i, j]),
                     horizontalalignment="center",
                     fontsize=18,
                     color="green" if cm_normalized[i, j] > thresh else "red")

    plt.tight_layout()
    plt.show()
    fig.savefig('%s/confusion_matrix.png' % (configuration['exp_folder']))

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

    # Load attributes
    attributes_data = np.loadtxt(configuration['embedding_attributes'])
    attribute_data = np.loadtxt(configuration['embedding_attributes'][:-4] + 's.txt')
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

    if configuration['debug']:
        X_train = cnn_data[id_train_samples][:500]
        A_train = attributes_data[id_train_samples][:500]
        Y_train = id_labels[id_train_samples][:500]
    else:
        X_train = cnn_data[id_train_samples]
        A_train = attributes_data[id_train_samples]
        Y_train = id_labels[id_train_samples]

    X_test = cnn_data[id_test_samples]
    A_test = attributes_data[id_test_samples]
    Y_test = id_labels[id_test_samples]

    labels_test = np.unique(Y_test)
    knn_feat = attribute_data[labels_test]

    return X_train, A_train, Y_train, X_test, A_test, Y_test, knn_feat, labels_test


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


def save_model(clf):
    file_config = configuration['exp_folder'] + 'model.json'
    file_weight = configuration['exp_folder'] + 'model_weights.h5'
    model_json = clf.to_json()
    with open(file_config, "w") as json_file:
        json_file.write(model_json)
    clf.save_weights(file_weight)
    print("Saved model to disk")


def euclidean_distance(y_true, y_pred):
    from keras.optimizers import K
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=1))

def euclidean_distance2(y_true, y_pred):
    from keras.optimizers import K
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=0))

def distance(y_true, y_pred):
    return np.sqrt(np.sum(np.square(y_true - y_pred), axis=1))

# TODO: find K function to repeat tensorflow

def function_M(C, y_pred, Di, W_weight):
    from keras.optimizers import K
    #x = np.repeat(y_pred, W_weight.shape[1], axis=1)
    x = y_pred.repeat(W_weight.shape[0])
    #Dt = np.array([euclidean_distance(y_pred, w) for w in W_weight])
    Dt = euclidean_distance(x, W_weight)
    return 0.5 * K.sum(C + (0.5 * Di) - (0.5 * Dt))


alpha = 0.6
def sigal_loss(input):
    def loss(y_true, y_pred):
        from keras.optimizers import K
        Ws, Wt, C = input
        Ws = np.delete(Ws, np.where(Ws == y_true)[0], axis=0)
        Di = euclidean_distance(y_true, y_pred)
        Ms = function_M(C, y_pred, Di, Ws)
        Mt = function_M(C, y_pred, Di, Wt)

        return (alpha * Di) + (1 - alpha)*(Ms + Mt)
    return loss


def get_semantic_embedding():

    attribute_data = np.loadtxt(configuration['embedding_attributes'][:-4] + 's.txt')
    train_classes = np.loadtxt(configuration['dataset'] + 'features/train_classes.txt').astype(np.int) - 1
    test_classes = np.loadtxt(configuration['dataset'] + 'features/train_classes.txt').astype(np.int) - 1

    return attribute_data[train_classes], attribute_data[test_classes]

def build_model():
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.regularizers import l2
    from keras.optimizers import SGD

    input_x = Input(shape=(4096,), name='input_x')

    f_x = Dense(300,
                # W_regularizer=l2(configuration['lamda_regularizer']),
                name='W_embedding',
                activation='linear')(input_x)

    sgd = SGD(lr=configuration['lr'], decay=configuration['lr_decay'], momentum=.9)
    model = Model(input=input_x, output=f_x)

    Ws, Wt = get_semantic_embedding()

    model.compile(optimizer='sgd',
                  loss=sigal_loss((Ws, Wt, 0.01)))
    return model

# def penalized_loss(noise):
#     def loss(y_true, y_pred):
#         from keras.optimizers import K
#         return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise), axis=-1)
#     return loss
# input1 = Input(batch_shape=(batch_size, timesteps, features))
# lstm =  LSTM(features, stateful=True, return_sequences=True)(input1)
# output1 = TimeDistributed(Dense(features, activation='sigmoid'))(lstm)
# output2 = TimeDistributed(Dense(features, activation='sigmoid'))(lstm)
# model = Model(input=[input1], output=[output1, output2])
# model.compile(loss=[penalized_loss(noise=output2), penalized_loss(noise=output1)], optimizer='rmsprop')


def save_exp_config():
    print '#' * 50, '\n'
    print 'Configuration: \n', json.dumps(configuration, sort_keys=True, indent=4)
    print '#' * 50, '\n'
    print '-' * 50, '\nLoading data ...\n', '-' * 50
    print '\n', '#' * 50

    with open(configuration['exp_folder'] + 'exp_setup.json', 'w') as outfile:
        obj_ = {'configuration': configuration}
        json.dump(obj_, outfile, sort_keys=True, indent=4)


def main(args):


    save_exp_config()

    X_train, A_train, Y_train, \
    X_test, A_test, Y_test, \
    knn_feat, labels_test = collect_splits(configuration['dataset'] + 'features/')

    with open('./data/awa/features/tmp/ids_classname.json') as outp:
        id_class = json.load(outp)

    # Embedding model
    print '-' * 50, '\nTraining embedding model\n', '-' * 50
    clf = build_model()
    start = time.time()

    #remote = visual_mode()

    #from keras.callbacks import EarlyStopping
    #earlier_stop = EarlyStopping(monitor='val_loss', patience=2000, verbose=1, mode='min')

    filepath_ = configuration['exp_folder'] + 'weights/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath_, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True,
                                 mode='min', period=1000)


    ######################################################################
    # SUPERVISED LEARNING                                                #
    ######################################################################

    history = clf.fit(X_train, A_train,
                      validation_split=0.1,
                      shuffle=False,
                      nb_epoch=configuration['number_epochs'],
                      callbacks=[checkpoint])

    ######################################################################
    # EVALUATION                                                         #
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

    with open(configuration['exp_folder'] + 'history.json', 'w') as outfile:
        obj_ = {'history': history.history}
        json.dump(obj_, outfile, sort_keys=True, indent=4)


    print 'Time (load training): %f' % (time.time() - start)
    print '-' * 50, '\nSaving embedding model\n', '-' * 50

    save_model(clf)

    print '-' * 50, '\nEmbedding model Trained\n', '-' * 50

    evaluation['score_train'] = clf.evaluate(X_train, A_train)
    evaluation['score_test'] = clf.evaluate(X_test, A_test)

    ######################################################################
    # ZERO-SHOT LEARNING                                                 #
    ######################################################################

    # embedding features from supervised model
    embd_test = clf.predict(X_test)

    # saving embedding
    np.savetxt(configuration['exp_folder'] + 'embd_test.txt', embd_test, fmt='%f')


    print '-' * 50, '\nDistance model prepared\n', '-' * 50
    knn = KNeighborsClassifier(n_neighbors=configuration['#neighbors'], metric='euclidean')
    knn.fit(knn_feat, labels_test)

    # Apply distance model
    print '-' * 50, '\nRunning Zero-Shot Learning\n', '-' * 50
    zsl_test = knn.predict(embd_test)

    zsl_scores, zsl_pred = knn.kneighbors(embd_test, n_neighbors=len(labels_test), return_distance=True)

    # Converting position to labels
    rows,cols = zsl_pred.shape
    for row in range(rows):
        for col in range(cols):
            zsl_pred[row,col] = labels_test[zsl_pred[row,col]]

    from utils.roc import plot_roc, get_roc_data

    ######################################################################
    # PLOT ROC                                                           #
    ######################################################################

    results = get_roc_data(y_gt=Y_test, y_pred=zsl_pred, y_scores=zsl_scores, labels=labels_test)
    plot_roc(results, id_class, configuration['exp_folder'])

    print '-' * 50, '\nRunning Evaluation\n', '-' * 50
    evaluation['accuracy_test'] = metrics.accuracy_score(Y_test,
                                                         zsl_test)
    evaluation['precision_test'] = metrics.precision_score(Y_test,
                                                           zsl_test,
                                                           labels=labels_test,
                                                           average='weighted')
    evaluation['recall_test'] = metrics.recall_score(Y_test,
                                                     zsl_test,
                                                     labels=labels_test,
                                                     average='weighted')

    cm_ = metrics.confusion_matrix(Y_test,
                                   zsl_test,
                                   labels=labels_test)
    np.savetxt(configuration['exp_folder'] + '/confusion_matrix.txt', cm_, fmt='%d')

    ######################################################################
    # PLOT CONFUSION MATRIX                                              #
    ######################################################################

    print_classes = [[id_class[str(label+1)] for label in labels_test]]
    plot_confusion_matrix(cm_, print_classes)

    print '-' * 50, '\nRunning Evaluation per Class\n', '-' * 50

    eval_per_class = {}
    for c in labels_test:
        acc_ = metrics.accuracy_score(Y_test == c,
                                      zsl_test == c)

        pr_ = metrics.precision_score(Y_test == c,
                                      zsl_test == c)

        re_ = metrics.recall_score(Y_test == c,
                                   zsl_test == c)

        eval_per_class[c] = {
                             'label': id_class[str(c+1)],
                             'accuracy': acc_,
                             'precision': pr_,
                             'recall': re_
                             }

    evaluation['~evaluation_per_class'] = eval_per_class

    print json.dumps(evaluation, sort_keys=True, indent=4)

    with open(configuration['exp_folder'] + 'evaluation.json', 'w') as outfile:
        obj_ = {'evaluation': evaluation}
        json.dump(obj_, outfile, sort_keys=True, indent=4)

    plt.show()

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

    evaluation, clf = main(args)

    plt.clf()

    print 'Total (time): %f' % (time.time() - t_global)

    print '---\nClosing application ...\n'
