"""
Classifer:
1) MultinomialNB(alpha=1.0)
2) BernoulliNB(alpha=1.0)
3) svm.SVC(kernel="linear", C=1)
4) linear_model.LogisticRegression(C=0.8)
"""
from __future__ import print_function
import time
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn import linear_model
from classifier import load_train_data, Classifier


def print_title(title, symbol="*", counts=30):
    print(symbol * counts)
    print(title)
    print(symbol * counts)


def mnbc_func(train_x, train_y):
    print("Dimension of input train_x: ", train_x.shape)
    alphas = [0, 1, 5, 10, 20, 100]
    results = []
    for a in alphas:
        t1 = time.time()
        clf = MultinomialNB(alpha=a)
        classifier = Classifier(clf, train_x, train_y)
        score, score_std, score_type = classifier.cross_validation(nfolds=10)
        results.append(score)
        t2 = time.time()
        print("Finshed training MNBC(alpha: %.1f) at: %.1f sec --- "
              "score: %.5f(%s)" % (a, t2-t1, score, score_type))


def test_mnbc():
    print_title("Multinomial NBC")
    # percentile_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    percentile_list = [40]
    for perc in percentile_list:
        print("=" * 20)
        print("Feature selection percentile: %d" % perc)
        train_data = load_train_data(
            "./data/train.txt", word_count_threshold=2, freq=True,
            feature_selection_flag=True, percentile=perc)
        mnbc_func(train_data['x'], train_data['y'])


def bnbc_func(train_x, train_y):
    print("Dimension of input train_x: ", train_x.shape)
    alphas = [0, 1, 5, 10, 20, 100]
    results = []
    for a in alphas:
        t1 = time.time()
        clf = BernoulliNB(alpha=a)
        classifier = Classifier(clf, train_x, train_y)
        score, score_std, score_type = classifier.cross_validation(nfolds=10)
        results.append(score)
        t2 = time.time()
        print("Finshed training BNBC(alpha: %.1f) at: %.1f sec --- "
              "score: %.5f(%s)" % (a, t2-t1, score, score_type))


def test_bnbc():
    print_title("Binomial NBC")
    #percentile_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    percentile_list = [40]
    for perc in percentile_list:
        print("=" * 20)
        print("Feature selection percentile: %d" % perc)
        train_data = load_train_data(
            "./data/train.txt", word_count_threshold=2, freq=False,
            feature_selection_flag=True, percentile=perc)
        bnbc_func(train_data["x"], train_data["y"])


def svm_func(train_x, train_y, kernel="linear"):
    print("Dimension of input train_x: ", train_x.shape)
    #c_values = [1, 10, 100, 1000]
    c_values = [1, 10]
    results = []
    for c in c_values:
        t1 = time.time()
        clf = svm.SVC(kernel=kernel, C=c)
        classifier = Classifier(clf, train_x, train_y)
        score, score_std, score_type = classifier.cross_validation(nfolds=10)
        results.append(score)
        t2 = time.time()
        print("Finshed training SVM(c: %.1f) at: %.1f sec --- score: %.5f(%s)"
              % (c, t2-t1, score, score_type))


def test_svm(kernel):
    freq = False
    print_title("SVM Classifier")
    percentile_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #percentile_list = [40]
    for perc in percentile_list:
        print("=" * 20 + "\nkernel: %s || freq: %s" % (kernel, freq))
        print("Feature selection percentile: %d" % perc)
        train_data = load_train_data(
            "./data/train.txt", word_count_threshold=1, freq=freq,
            feature_selection_flag=True, percentile=perc)
        svm_func(train_data['x'], train_data['y'], kernel=kernel)


def entropy_func(train_x, train_y):
    print("Dimension of input train_x: ", train_x.shape)
    c_values = [0.1, 1, 10, 100]
    results = []
    for c in c_values:
        t1 = time.time()
        clf = linear_model.LogisticRegression(C=c)
        classifier = Classifier(clf, train_x, train_y)
        score, score_std, score_type = classifier.cross_validation(nfolds=10)
        results.append(score)
        t2 = time.time()
        print("Finshed training MaxEntropy(c: %.1f) at: %.1f sec --- "
              "score: %.5f(%s)" % (c, t2-t1, score, score_type))


def test_entropy():
    print_title("Max-Entropy Classifier")
    percentile_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #percentile_list = [40]
    for perc in percentile_list:
        print("=" * 20)
        print("Feature selection percentile: %d" % perc)
        train_data = load_train_data(
            "./data/train.txt", word_count_threshold=1, freq=False,
            feature_selection_flag=True, percentile=perc)
        entropy_func(train_data['x'], train_data['y'])


if __name__ == "__main__":
    #test_mnbc()
    #test_bnbc()
    # test_entropy()
    #test_svm("rbf")
    test_svm("linear")
