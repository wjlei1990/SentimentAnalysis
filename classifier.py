from __future__ import print_function, division
import time
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import confusion_matrix, classification_report, \
    roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn import svm
from preprocessSentences2 import tokenize_corpus, find_wordcounts, \
    wordcount_filter


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('%s function took %s sec' % (f.func_name, (time2-time1)))
        return ret
    return wrap


def turn_freq_into_pres(data):
    """ Turn frequency into presense by clip the data """
    return data.clip(0, 1)


def feature_selection(x, y, percentile=60):
    """ Input feature selection(dimension reduction) """
    selector = SelectPercentile(chi2, percentile=percentile)
    selector.fit(x, y)
    x_new = selector.transform(x)
    indexs = selector.get_support(indices=True)
    return x_new, indexs


def append_punc_to_train_x(train_x, punc_list, punc_count):
    nsamples = train_x.shape[0]
    nfeatures = train_x.shape[1]
    train_x_new = np.zeros(
        [nsamples, nfeatures + len(punc_count)])

    # copy old values
    train_x_new[:, 0:nfeatures] = train_x
    # copy punc
    for idx in range(len(punc_list)):
        train_x_new[:, nfeatures + idx] = punc_count[punc_list[idx]]

    return train_x_new


def load_train_data(train_file, word_count_threshold=5, freq=False,
                    feature_selection_flag=True, percentile=60, keep_neg=False,
                    punc_flag=True, punc_list=["!", "?"]):
    """ Load train data and class """
    docs, classes, _, words, punc_counts = tokenize_corpus(
         train_file, keep_neg=keep_neg, train=True, punc_list=punc_list)
    # turn classes into int
    train_y = np.array([int(x) for x in classes])
    # prepare vocab and bag_of_words
    vocab = wordcount_filter(words, num=word_count_threshold)
    train_x = find_wordcounts(docs, vocab)

    if not freq:
        train_x = turn_freq_into_pres(train_x)

    if punc_flag:
        train_x = append_punc_to_train_x(train_x, punc_list, punc_counts)

    vocab_len = len(vocab)
    punc_list_new = []
    if feature_selection_flag:
        train_x, indexs = feature_selection(
            train_x, train_y, percentile=percentile)
        # re-select vocab
        vocab_new = [vocab[idx] for idx in indexs if idx < vocab_len]
        punc_list_new = [punc_list[idx-vocab_len] for idx in indexs
                         if idx >= vocab_len]
        vocab = vocab_new
        punc_list = punc_list_new

    print("punc list: %s" % punc_list)

    # if not freq:
    #    train_x = turn_freq_into_pres(train_x)

    return {"x": train_x, "y": train_y, "vocab": vocab, "docs": docs,
            "punc_list": punc_list}


def load_test_data(test_file, vocab, freq=False, keep_neg=False,
                   punc_flag=True, punc_list=["!", "?"]):
    docs, classes, samples, punc_counts = tokenize_corpus(
        test_file, keep_neg=keep_neg, train=False, punc_list=punc_list)
    # transfer string into int
    test_y = [int(x) for x in classes]
    test_x = find_wordcounts(docs, vocab)

    if not freq:
        test_x = turn_freq_into_pres(test_x)

    if punc_flag:
        test_x = append_punc_to_train_x(test_x, punc_list, punc_counts)

    return {"x": test_x, "y": test_y, "docs": docs}


class Classifier(object):

    def __init__(self, clf, train_x, train_y):
        self.clf = clf
        self.train_x = train_x
        self.train_y = train_y

    def train(self):
        """
        Over-loading in child class
        """
        self.clf.fit(self.train_x, self.train_y)

    @timing
    def cross_validation(self, nfolds=10, scoring='accuracy'):
        """
        For scoring types, please check out the website at:
        http://scikit-learn.org/stable/modules/model_evaluation.html#
            scoring-parameter

        :param scoring: scoring type, could be:
            1) 'accuracy'
            2) 'f1_marco'
        :param scoring: str
        """
        scores = cross_val_score(
            self.clf, self.train_x, self.train_y, cv=nfolds,
            scoring=scoring)
        return scores.mean(), scores.std(), scoring

    def predict(self, test_x, test_y):
        predict_y = self.clf.predict(test_x)
        conf = confusion_matrix(test_y, predict_y)
        report = classification_report(test_y, predict_y, digits=3)
        return conf, report

    def get_y_score(self, test_x):
        if isinstance(self.clf, svm.SVC):
            y_scores = self.clf.decision_function(test_x)
        else:
            _scores = self.clf.predict_proba(test_x)
            y_scores = _scores[:, 1]
        return y_scores

    def roc_metric(self, test_x, test_y):
        y_scores = self.get_y_score(test_x)
        fpr, tpr, thresholds = roc_curve(test_y, y_scores, pos_label=1)
        score = roc_auc_score(test_y, y_scores)
        return {"fpr": fpr, "tpr": tpr, "score": score}

    def precision_recall_metric(self, test_x, test_y):
        y_scores = self.get_y_score(test_x)
        precision, recall, threshold = precision_recall_curve(
            test_y, y_scores, pos_label=1)
        score = auc(recall, precision)
        return {"precision": precision, "recall": recall, "score": score}
