import argparse
from sklearn import linear_model, svm, neural_network
import numpy as np
import pickle


def train(X, y, model='logistic', feed_dict={}):
    """Trains specified classification model on given data

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target classes ((n_samples,))
        model (str, optional): Model choice. Defaults to 'logistic'. \
            choose from ['logistic', 'SVM', 'MLP']
        feed_dict (dict, optional): extra arguments for model. Defaults to {}.
    """

    models = {
        'logistic': linear_model.LogisticRegression,
        'SVM': svm.LinearSVC,
        'MLP': neural_network.MLPClassifier
    }

    if model == 'logistic':
        feed_dict.update({'n_jobs': -1})

    classifier = models[model](**feed_dict)
    classifier.fit(X, y)

    print('Saving trained model ...')
    path = f'{model}-model.pkl'
    pickle.dump(classifier, open(path, 'wb'))
    print(f'Model successfully saved to {path}')

    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=False, default='logistic')
    ap.add_argument("--feed-dict", required=False, default=dict())
    av = ap.parse_args()

    train_X = np.load('../data/trainX.npy')
    train_y = np.load('../data/trainy.npy')
    
    feed_dict = eval(av.feed_dict) if av.feed_dict else dict()

    assert isinstance(feed_dict, dict), 'feed_dict must be of dictionary type'

    model = train(train_X, train_y, av.model, feed_dict)
