#!/usr/bin/env python

"""

sample:
python train.py -exp 'test001' >> ./experiments/awa/test_001.txt


"""
__author__ = "Rafael Felix"
__license__ = "UoA"
__copyright__ = "Copyright 2016"
__maintainer__ = "Rafael Felix"
__email__ = "rfelixmg@gmail.com"
__status__ = "beta"

import random, os, sys, argparse
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

import json, datetime, time, h5py
import numpy as np

from utils import file_utils
from modules.SigalLoss import SigalLoss

sys.path.append(os.path.abspath('..'))
evaluation = {}


def load_configuration(tag_name = 'awa'):

    dataset_ = './data/%s/' % tag_name

    configuration = {
        'dataset': dataset_,
        'dataset_image': dataset_ + 'images/',
        'dataset_text': dataset_ + 'fine_grained_description/',
        'dataset_attributes': dataset_ + 'attributes/',
        'embedding': dataset_ + 'features/',
        # 'embedding_image':          dataset_ + 'features/halah_googlenet/feature.txt',
        'embedding_image': dataset_ + 'features/lampert_vgg/feature.h5',
        'visual_input': 4096,
        'embedding_text': dataset_ + 'features/bow_text/None',
        # 'embedding_attributes':     dataset_ + 'features/bow_attributes/feature.txt',
        # 'embedding_attributes': dataset_ + 'features/word2vec/feature.txt',
        'embedding_attributes': dataset_ + 'features/w2v/feature.txt',
        'semantic_input': 100,
        #'embedding_attributes': dataset_ + 'features/glove/feature.txt',
        # 'estimation_attributes': dataset_ + '/attributes/class_attribute_labels_continuous.txt',
        #   mlp,
        'retrain': False,
        #
        'number_epochs': 100,
        'lr': 0.9,
        'lr_decay': 6e-4,
        'lamda_regularizer': 1e-5,
        #
        '#classes': None,
        'baseline_model': 'mlp',
        '#neighbors': 1,
        'save_every_epoch': 100,
        'n_estimators': 100,
        'max_iter': 200,
        'n_jobs': -2,
        'estimated_values': False,
        'output_file': '',
        'tag': tag_name,
        #
        'C': 1e-6,
        'alpha': 0.6,
        'margin': 0.01,
        #
        'experiment_name': 'test_000_',
        'resume': False,
        'debug': False,
        'fake_data': False,
        'process_per_epoch': False
    }

    if configuration['tag'] == 'cub':
        configuration['#classes'] = 200
    elif configuration['tag'] == 'awa':
        configuration['#classes'] = 50

    return configuration


def load_args(configuration):

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

    parser.add_argument('-r', '-resume', default=configuration['resume'],
                        help='Resume training from previous experiment', required=False)

    parser.add_argument('-ne', '-number_epochs', default=configuration['number_epochs'],
                        help='Number of epochs', required=False)

    parser.add_argument('-lr', '-learning_rate', default=configuration['lr'],
                        help='learning rate', required=False)

    parser.add_argument('-rt', '-re_train', default=configuration['retrain'],
                        help='Re-Train the network', required=False)

    parser.add_argument('-verbose', default=False,
                        help='Verbose debug mode', required=False)

    args = vars(parser.parse_args())


    if args['r']:
        configuration['resume'] = args['r'] + 'exp_setup.json'
        json_data = open(configuration['resume']).read()
        configuration = json.loads(json_data)['configuration']
        configuration['resume'] = args['r'] + 'exp_setup.json'
        configuration['exp_folder'] = args['r']
        f_name = 'exp_setup_%s.json' % datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        configuration['resume'] = args['r']
        configuration['experiment_name'] = args['exp'] + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        configuration['exp_folder'] = './experiments/%s/%s/' % (configuration['tag'], configuration['experiment_name'])

        configuration['output_file'] = '%s%s_evaluation_%dnn%s.txt' % \
                                       (configuration['exp_folder'],
                                        configuration['baseline_model'],
                                        configuration['#neighbors'],
                                        '_estimated_values' if configuration['estimated_values'] else '')

        file_utils.makedir(configuration['experiment_name'], './experiments/%s/' % configuration['tag'])
        file_utils.makedir('weights', configuration['exp_folder'])
        f_name = 'exp_setup.json'

    configuration['retrain'] = bool(args['rt'])
    configuration['lr'] = float(args['lr'])
    configuration['number_epochs'] = int(args['ne'])

    save_exp_config(f_name, configuration)

    return configuration


def plot_history(f_name, history):

    # summarize history for loss
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(111)
    ax.plot(history['loss'])
    ax.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show(block=False)
    fig.savefig(f_name)


def plot_confusion_matrix(cm_, classes, dirpath, normalize=True):

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
    plt.show(block=False)
    fig.savefig(dirpath)


def save_model(clf, name='model'):
    file_config = configuration['exp_folder'] + '%s.json' % name
    file_weight = configuration['exp_folder'] + '%s_weights.h5' % name
    model_json = clf.to_json()
    with open(file_config, "w") as json_file:
        json_file.write(model_json)
    clf.save_weights(file_weight)
    print("Saved model to disk")


def load_validation_data(Y, configuration):

    try:
        ids_train = np.loadtxt(configuration['dataset'] + 'tmp/ids_train.txt').astype(np.int)
        ids_valid = np.loadtxt(configuration['dataset'] + 'tmp/ids_valid.txt').astype(np.int)

        return ids_train, ids_valid
    except:
        from sklearn import cross_validation

        cv = cross_validation.ShuffleSplit(Y.shape[0], test_size=0.10, random_state=True)
        ids_train, ids_valid = next(iter(cv))

        np.savetxt(configuration['dataset'] + 'tmp/ids_train.txt', ids_train, fmt='%d')
        np.savetxt(configuration['dataset'] + 'tmp/ids_valid.txt', ids_valid, fmt='%d')

        return ids_train, ids_valid


def collect_splits(configuration):

    directory = configuration['dataset'] + 'features/'

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
        X_train = cnn_data[id_train_samples][:200]
        A_train = attributes_data[id_train_samples][:200]
        Y_train = id_labels[id_train_samples][:200]
    else:
        X_train = cnn_data[id_train_samples]
        A_train = attributes_data[id_train_samples]
        Y_train = id_labels[id_train_samples]

    X_test = cnn_data[id_test_samples]
    A_test = attributes_data[id_test_samples]
    Y_test = id_labels[id_test_samples]

    labels_test = np.unique(Y_test)
    knn_feat_test = attribute_data[labels_test]

    labels_train = np.unique(Y_train)
    knn_feat_train = attribute_data[labels_train]

    data_package = {'X_train': X_train,
                    'A_train': A_train,
                    'Y_train': Y_train,

                    'X_test': X_test,
                    'A_test': A_test,
                    'Y_test': Y_test,

                    'label_train': labels_train,
                    'feat_train': knn_feat_train,
                    'label_test': labels_test,
                    'feat_test': knn_feat_test,
                    }

    return data_package


def get_semantic_embedding(configuration):

    attribute_data = np.loadtxt(configuration['embedding_attributes'][:-4] + 's.txt')
    train_classes = np.loadtxt(configuration['dataset'] + 'features/train_classes.txt').astype(np.int) - 1
    test_classes = np.loadtxt(configuration['dataset'] + 'features/train_classes.txt').astype(np.int) - 1

    return attribute_data[train_classes], attribute_data[test_classes]

def build_knn(data, configuration):

    knn = KNeighborsClassifier(n_neighbors=configuration['#neighbors'], metric='euclidean')
    knn.fit(data['feat'], data['label'])
    return knn

def build_model(configuration):
    from keras.layers import Input, Dense, Merge, Lambda, merge, Flatten
    from keras.models import Model
    from keras.regularizers import l2
    from keras.optimizers import SGD, Adam
    from utils.experiments_util import get_fake_semantic_embedding

    def nothing_to_lose(y_true, y_pred):
        return y_pred

    input_x = Input(shape=(configuration['visual_input'],))
    input_u = Input(shape=(configuration['semantic_input'],))

    v_x = Dense(configuration['semantic_input'],
                   W_regularizer=l2(configuration['lamda_regularizer']),
                   name='W_embedding',
                   init='zero',
                   activation='linear')(input_x)

    # Loading persistent matrix
    if configuration['fake_data']:
        Ws, Wt = get_fake_semantic_embedding()
    else:
        Ws, Wt = get_semantic_embedding(configuration)

    sigal_layer = SigalLoss(Ws=Ws, Wt=Wt, C=configuration['C'],
                            alpha=configuration['alpha'],
                            margin=configuration['margin'])
    f_x = sigal_layer([v_x, input_u])

    model = Model(input=([input_x, input_u]), output=f_x)
    visual = Model(input=input_x, output=v_x)

    sgd = SGD(lr=configuration['lr'], decay=configuration['lr_decay'], nesterov=True)
    #sgd = SGD(lr=configuration['lr'], nesterov=True)
    #adam = SGD(lr=configuration['lr'], nesterov=True)

    model.compile(loss=nothing_to_lose, optimizer=sgd)

    return model, visual


def save_exp_config(f_name, configuration):
    print '#' * 50, '\n'
    print 'Configuration: \n', json.dumps(configuration, sort_keys=True, indent=4)
    print '#' * 50, '\n'
    print '-' * 50, '\nLoading data ...\n', '-' * 50
    print '\n', '#' * 50

    with open(configuration['exp_folder'] + f_name, 'w') as outfile:
        obj_ = {'configuration': configuration}
        json.dump(obj_, outfile, sort_keys=True, indent=4)


def train(data, clf, visual, configuration):

    print '-' * 50, '\nTraining embedding model\n', '-' * 50

    start = time.time()

    ######################################################################
    # CALLBACKS                                                          #
    ######################################################################
    from modules.Callbacks import ModelCheckpoint, ReportInformation
    filepath_ = configuration['exp_folder'] + 'weights/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath_, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True,
                                 mode='min', period=configuration['save_every_epoch'])

    lr_report = ReportInformation(f_name='/tmp/learning_rates.txt')

    ######################################################################
    # SUPERVISED LEARNING                                                #
    ######################################################################
    ids_train, ids_valid = load_validation_data(data['Y_train'], configuration)

    if configuration['process_per_epoch']:
        knn_train = build_knn({'feat': data['feat_train'], 'label': data['label_train']}, configuration=configuration)
        knn_test = build_knn({'feat': data['feat_test'], 'label': data['label_test']}, configuration=configuration)

        history = {'loss': [], 'val_loss':[], 'acc_train':[], 'acc_test':[]}

        fig_p = plt.figure(figsize=(30, 15))
        ax = fig_p.add_subplot(111)
        plt.axis((0, configuration['number_epochs'] + 1, 0, 1))

        for i in range(configuration['number_epochs']):
            h = clf.fit([data['X_train'][ids_train], data['A_train'][ids_train]],
                        data['Y_train'][ids_train],
                        validation_data=([data['X_train'][ids_valid], data['A_train'][ids_valid]],
                                         data['Y_train'][ids_valid]),
                        shuffle=True,
                        nb_epoch=1,
                        callbacks=[])
            history['loss'].append(h.history['loss'][0])
            history['val_loss'].append(h.history['val_loss'][0])

            embd_train = visual.predict(data['X_train'])
            acc_train = knn_train.score(embd_train, data['Y_train'])

            embd_test = visual.predict(data['X_test'])
            acc_test = knn_test.score(embd_test, data['Y_test'])

            print 'Epoch [%d]: ' % i
            print 'Learning rate: ', (clf.optimizer.lr * (1. / (1. + clf.optimizer.decay * i))).eval()
            print 'Accuracy Training: ', acc_train
            print 'Accuracy Testing: ', acc_test

            history['acc_train'].append(acc_train)
            history['acc_test'].append(acc_test)

            # TODO: Keep from here.
            if i % 10 == 0:
                # summarize history for loss
                ax.cla()
                ax.plot(history['acc_train'], np.arange(len(history['acc_train'])))
                ax.plot(history['acc_test'], np.arange(len(history['acc_test'])))

                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show(block=False)

            print '=+=+==' * 20


    else:
        history = clf.fit([data['X_train'][ids_train], data['A_train'][ids_train]],
                          data['Y_train'][ids_train],
                          validation_data=([data['X_train'][ids_valid], data['A_train'][ids_valid]],
                                           data['Y_train'][ids_valid]),
                          shuffle=True,
                          nb_epoch=configuration['number_epochs'],
                          callbacks=[] if configuration['debug'] else [checkpoint, lr_report])
        history = history.history
    plot_history('%s/loss_val_loss.png' % (configuration['exp_folder']), history)


    ######################################################################
    # Saving meta evaluation                                             #
    ######################################################################


    with open(configuration['exp_folder'] + 'history.json', 'w') as outfile:
        obj_ = {'history': history}
        json.dump(obj_, outfile, sort_keys=True, indent=4)

    print 'Time (load training): %f' % (time.time() - start)
    print '-' * 50, '\nSaving embedding model\n', '-' * 50

    save_model(clf, 'model')
    save_model(visual, 'visual')

    print '-' * 50, '\nEmbedding model Trained\n', '-' * 50

    return clf, visual


def run_zsl(visual, X, knn_feat, labels_test, configuration, aux_ = ''):
    embd_test = visual.predict(X)

    # saving embedding
    np.savetxt(configuration['exp_folder'] + '%sembd_test.txt' % aux_, embd_test, fmt='%f')


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

    return zsl_test, zsl_scores, zsl_pred


def evaluate_model(id_class, labels_test, zsl_pred, zsl_scores, Y_test, zsl_test, configuration, aux_=''):

    from utils.roc import plot_roc, get_roc_data
    ######################################################################
    # PLOT ROC                                                           #
    ######################################################################

    results = get_roc_data(y_gt=Y_test, y_pred=zsl_pred, y_scores=zsl_scores, labels=labels_test)
    plot_roc(results, id_class, configuration['exp_folder'] + '/%sroc_curve' % aux_)

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
    np.savetxt(configuration['exp_folder'] + '/%sconfusion_matrix.txt' % aux_, cm_, fmt='%d')

    ######################################################################
    # PLOT CONFUSION MATRIX                                              #
    ######################################################################

    print_classes = [[id_class[str(label + 1)] for label in labels_test]]
    plot_confusion_matrix(cm_, print_classes, dirpath='%s/%sconfusion_matrix.png' % (configuration['exp_folder'], aux_))

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
            'label': id_class[str(c + 1)],
            'accuracy': acc_,
            'precision': pr_,
            'recall': re_
        }

    evaluation['~evaluation_per_class'] = eval_per_class

    print json.dumps(evaluation, sort_keys=True, indent=4)

    with open(configuration['exp_folder'] + '%sevaluation.json' % aux_, 'w') as outfile:
        obj_ = {'evaluation': evaluation}
        json.dump(obj_, outfile, sort_keys=True, indent=4)

    plt.show(block=False)

    return evaluation

if __name__ == '__main__':

    from utils.experiments_util import collect_fake_splits

    print '\n\n\nInitializing application...\n\n'

    random.seed(0)

    configuration = load_args(load_configuration('awa'))

    t_global = time.time()

    if configuration['fake_data']:
        data_package = collect_fake_splits(200)
        id_class = {str(i+1): i+1 for i in range(50)}

    else:
        data_package = collect_splits(configuration)

        with open('./data/awa/features/tmp/ids_classname.json') as outp:
            id_class = json.load(outp)

    clf, visual = build_model(configuration)

    if configuration['resume']:
        print '=' * 45
        print 'Resume previous experiment'
        print '=' * 45

        try:
            clf.load_weights(configuration['exp_folder'] + 'model_weights.h5')
            print '###Importing weights from h5'
        except:
            import h5py
            print '###Importing weights from h5py'
            f = h5py.File(configuration['exp_folder'] + 'model_weights.hdf5', 'r')
            clf.load_weights_from_hdf5_group(f)

        if configuration['retrain']:
            print '*' * 45
            print 'Re-Training the network'
            print '*' * 45
            clf, visual = train(data=data_package, clf=clf, visual=visual, configuration=configuration)
    else:
        clf, visual = train(data=data_package, clf=clf, visual=visual, configuration=configuration)

    # Test
    zsl_test, zsl_scores, zsl_pred = run_zsl(visual, data_package['X_test'], data_package['feat_test'],
                                             data_package['label_test'], configuration=configuration)
    evaluation = evaluate_model(id_class, data_package['label_test'], zsl_pred, zsl_scores,
                                data_package['Y_test'], zsl_test, configuration=configuration)

    # Train
    zsl_test, zsl_scores, zsl_pred = run_zsl(visual, data_package['X_train'], data_package['feat_train'],
                                             data_package['label_train'], configuration=configuration)
    evaluation = evaluate_model(id_class, data_package['label_train'], zsl_pred, zsl_scores,
                                data_package['Y_train'], zsl_test, configuration=configuration)

    plt.show(block=True)
    plt.clf()

    print 'Total (time): %f' % (time.time() - t_global)

    print '---\nClosing application ...'
