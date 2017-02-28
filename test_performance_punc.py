"""
Classifer:
1) MultinomialNB(alpha=1.0)
2) BernoulliNB(alpha=1.0)
3) svm.SVC(kernel="linear", C=1)
4) linear_model.LogisticRegression(C=0.8)
"""
from __future__ import print_function
from copy import deepcopy
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plt
from classifier import load_train_data, Classifier, load_test_data


def print_title(title, symbol="*", counts=20):
    print(symbol * counts)
    print(title)
    print(symbol * counts)


def get_train_data(perc, freq):
    print("Feature selection percentile: %d" % perc)
    train_data = load_train_data(
        "./data/train.txt", word_count_threshold=2, freq=freq,
        feature_selection_flag=True, percentile=perc,
        keep_neg=False, punc_flag=True)
    return train_data


def get_test_data(vocab, punc_list, freq):
    test_data = load_test_data(
        "./data/test.txt", vocab, freq=freq,
        keep_neg=False, punc_flag=True, punc_list=punc_list)
    return test_data


def test_mnbc(train_data, test_data):
    print_title("Multinomial NBC")
    print("Dimension of input train_x: ", train_data['x'].shape)
    clf = MultinomialNB(alpha=1.0)
    classifier = Classifier(clf, train_data['x'], train_data['y'])
    classifier.train()
    conf, report = classifier.predict(test_data['x'], test_data['y'])
    print("Confusion matrix:")
    print(conf)
    print(report)

    roc_metric = classifier.roc_metric(test_data['x'], test_data['y'])
    print("roc score: %f" % roc_metric["score"])

    pr_metric = classifier.precision_recall_metric(
        test_data['x'], test_data['y'])
    print("pr score: %f" % pr_metric["score"])

    return roc_metric, pr_metric


def test_bnbc(train_data, test_data):
    print_title("Binomial NBC")
    print("Dimension of input train_x: ", train_data['x'].shape)
    clf = BernoulliNB(alpha=1.0)
    classifier = Classifier(clf, train_data['x'], train_data['y'])
    classifier.train()
    conf, report = classifier.predict(test_data['x'], test_data['y'])
    print("Confusion matrix:")
    print(conf)
    print(report)

    roc_metric = classifier.roc_metric(test_data['x'], test_data['y'])
    print("roc score: %f" % roc_metric["score"])

    pr_metric = classifier.precision_recall_metric(
        test_data['x'], test_data['y'])
    print("pr score: %f" % pr_metric["score"])

    return roc_metric, pr_metric


def test_svm(train_data, test_data, kernel="linear"):
    print_title("SVM Classifier(%s)" % kernel)
    print("Dimension of input train_x: ", train_data['x'].shape)

    if kernel == "linear":
        clf = svm.SVC(kernel=kernel, C=1.0)
    elif kernel == "rbf":
        clf = svm.SVC(kernel=kernel, C=100.0)

    classifier = Classifier(clf, train_data['x'], train_data['y'])
    classifier.train()
    conf, report = classifier.predict(test_data['x'], test_data['y'])
    print("Confusion matrix:")
    print(conf)
    print(report)

    roc_metric = classifier.roc_metric(test_data['x'], test_data['y'])
    print("roc score: %f" % roc_metric["score"])

    pr_metric = classifier.precision_recall_metric(
        test_data['x'], test_data['y'])
    print("pr score: %f" % pr_metric["score"])

    return roc_metric, pr_metric


def test_entropy(train_data, test_data):
    print_title("Max-Entropy Classifier")
    print("Dimension of input train_x: ", train_data['x'].shape)
    clf = linear_model.LogisticRegression(C=1.0)
    classifier = Classifier(clf, train_data['x'], train_data['y'])
    classifier.train()
    conf, report = classifier.predict(test_data['x'], test_data['y'])
    print("Confusion matrix:")
    print(conf)
    print(report)

    roc_metric = classifier.roc_metric(test_data['x'], test_data['y'])
    print("roc score: %f" % roc_metric["score"])

    pr_metric = classifier.precision_recall_metric(
        test_data['x'], test_data['y'])
    print("pr score: %f" % pr_metric["score"])

    return roc_metric, pr_metric


def plot_roc_curve(roc_metric):
    plt.figure(figsize=(5, 4))
    keys = ["mnbc", "bnbc", "entropy", "svm_linear", "svm_rbf"]
    labels = {"mnbc": "mnbc", "bnbc": "bnbc", "entropy": "LR",
              "svm_linear": "SVML", "svm_rbf": "SVM-rbf"}
    for k in keys: 
        v = roc_metric[k]
        x = v["fpr"]
        y = v["tpr"]
        plt.plot(x, y, label="%s(auc=%.3f)" % (labels[k], v["score"]))

    plt.plot([0, 1.0], [0.0, 1.0], "b--", alpha=0.8)
    # plt.title("ROC Curves")
    plt.xlabel("Specificity")
    plt.ylabel("Sensitivity")
    plt.grid()
    plt.legend(loc="lower right")
    plt.tight_layout()
    # plt.show()
    plt.savefig("roc_curve_chi2.pdf")


def plot_pr_curve(pr_metric):
    plt.figure(figsize=(5, 4))

    keys = ["mnbc", "bnbc", "entropy", "svm_linear", "svm_rbf"]
    labels = {"mnbc": "mnbc", "bnbc": "bnbc", "entropy": "LR",
              "svm_linear": "SVML", "svm_rbf": "SVM-rbf"}
    for k in keys:
        v = pr_metric[k]
        x = v["recall"]
        y = v["precision"]
        plt.plot(x, y, label="%s(auc=%.3f)" % (labels[k], v["score"]))

    plt.plot([0, 1.0], [0.5, 0.5], "b--", alpha=0.8)
    # plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.legend(loc="lower left")
    plt.tight_layout()
    #plt.show()
    plt.savefig("pr_curve_chi2.pdf")


if __name__ == "__main__":
    roc_metric = {}
    pr_metric = {}
    # frequency method
    train_data = get_train_data(40, True)
    print("train -- x, y, vocab, doc:", train_data['x'].shape,
          len(train_data['y']), len(train_data['vocab']),
          len(train_data['docs']))
    test_data = get_test_data(
        train_data['vocab'], train_data["punc_list"], True)
    print("test  -- x, y, doc:", test_data['x'].shape, len(test_data['y']),
          len(test_data['docs']))
    roc_metric["mnbc"], pr_metric["mnbc"] = \
        test_mnbc(deepcopy(train_data), deepcopy(test_data))

    # presence method
    train_data = get_train_data(40, False)
    test_data = get_test_data(
        train_data['vocab'], train_data["punc_list"], False)
    roc_metric["bnbc"], pr_metric["bnbc"] = \
        test_bnbc(deepcopy(train_data), deepcopy(test_data))
    roc_metric["entropy"], pr_metric["entropy"] = \
        test_entropy(deepcopy(train_data), deepcopy(test_data))
    roc_metric["svm_rbf"], pr_metric["svm_rbf"] = \
        test_svm(deepcopy(train_data), deepcopy(test_data), kernel="rbf")
    roc_metric["svm_linear"], pr_metric["svm_linear"] = \
        test_svm(deepcopy(train_data), deepcopy(test_data), kernel="linear")

    plot_roc_curve(roc_metric)
    plot_pr_curve(pr_metric)
