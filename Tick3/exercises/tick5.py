from typing import List, Dict, Union
import random
import os, sys
sys.path.append('/Users/yangzejia/Desktop/PartIA/mlrd/Tick3')
from utils.sentiment_detection import read_tokens, load_reviews, print_binary_confusion_matrix
from exercises.tick1 import accuracy, read_lexicon, predict_sentiment
from exercises.tick2 import predict_sentiment_nbc, calculate_smoothed_log_probabilities, \
    calculate_class_log_probabilities
from tick4 import read_lexicon_magnitude, predict_sentiment_magnitude
#from tick4 import read_lexicon_magnitude, predict_sentiment_magnitude


def generate_random_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    random.shuffle(training_data)
    fold_size = len(training_data) // n
    training_folds = []
    for i in range(n):
        start = i * fold_size
        end = (i + 1) * fold_size if (i + 1) * fold_size <= len(training_data) else len(training_data)
        training_folds.append(training_data[start : end])
    return training_folds


def generate_stratified_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, stratified.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    random.shuffle(training_data)
    pos = []
    neg = []
    for review in training_data:
        if review['sentiment'] == 1:
            pos.append(review)
        else:
            neg.append(review)
    training_st_fold = []

    pfold = len(pos) // n
    nfold = len(neg) // n
    for i in range(n):
        ps = i * pfold
        pe = (i + 1) * pfold if (i + 1) * pfold <= len(pos) else len(pos)

        ns = i * nfold
        ne = (i + 1) * nfold if (i + 1) * nfold <= len(neg) else len(neg)

        training_st_fold.append(pos[ps:pe] + neg[ns:ne])
    
    return training_st_fold


def cross_validate_nbc(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        training_data = []
        for j in range(1, n):
            training_data += split_training_data[(i + j) % n]
        class_log_probabilities = calculate_class_log_probabilities(training_data)
        log_probabilities = calculate_smoothed_log_probabilities(training_data)

        accuracy = []
        validation_data = split_training_data[i]
        for review in validation_data:
            predict = predict_sentiment_nbc(review['text'], log_probabilities, class_log_probabilities)
            accuracy.append(predict == review['sentiment'])
        accuracies.append(sum(accuracy) / len(accuracy))

    return accuracies

def cross_validation_accuracy(accuracies: List[float]) -> float:
    """Calculate the mean accuracy across n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: mean accuracy over the cross folds
    """
    return sum(accuracies) / len(accuracies)


def cross_validation_variance(accuracies: List[float]) -> float:
    """Calculate the variance of n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: variance of the cross fold accuracies
    """
    mean = cross_validation_accuracy(accuracies)
    sum_var = 0
    for acc in accuracies:
        sum_var += (acc - mean) ** 2
    return sum_var / len(accuracies)


def confusion_matrix(predicted_sentiments: List[int], actual_sentiments: List[int]) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for pre, act in zip(predicted_sentiments, actual_sentiments):
        if act == 1:
            if pre == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pre == -1:
                tn += 1
            else:
                fp += 1
    return [[tp, fp], [fn, tn]]
    

def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # First test cross-fold validation
    folds = generate_random_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Random cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Random cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Random cross validation variance: {variance}\n")

    folds = generate_stratified_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    #print(accuracies)
    print(f"Stratified cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Stratified cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Stratified cross validation variance: {variance}\n")

    # Now evaluate on 2016 and test
    '''
    class_priors = calculate_class_log_probabilities(tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(tokenized_data)

    preds_test = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    for review in test_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_test.append(pred)

    acc_smoothed = accuracy(preds_test, test_sentiments)
    print(f"Smoothed Naive Bayes accuracy on held-out data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test, test_sentiments))

    preds_recent = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_recent.append(pred)

    acc_smoothed = accuracy(preds_recent, recent_sentiments)
    print(f"Smoothed Naive Bayes accuracy on 2016 data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent, recent_sentiments))

    # evaluate simple classifier on 2016 dataset
    
    tokens = [test_tokens, recent_tokens]
    labels = [test_sentiments, recent_sentiments]
    dataset = ['held-out', '2016']
    
    lexicon_magnitude = read_lexicon_magnitude(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))
    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    for tokens, sentiments, data in zip(tokens, labels, dataset):
        preds_magnitude = []
        preds_simple = []
        for review in tokens:
            pred = predict_sentiment_magnitude(review, lexicon_magnitude)
            preds_magnitude.append(pred)
            pred_simple = predict_sentiment(review, lexicon)
            preds_simple.append(pred_simple)

        acc_magnitude = accuracy(preds_magnitude, sentiments)
        acc_simple = accuracy(preds_simple, sentiments)

        print(f"simple classifier accuracy on {data} data: {acc_simple}")
        print("Confusion matrix:")
        print_binary_confusion_matrix(confusion_matrix(preds_simple, sentiments))

        print(f"simple classifier with magnitude accuracy on {data} data: {acc_magnitude}")
        print("Confusion matrix:")
        print_binary_confusion_matrix(confusion_matrix(preds_magnitude, sentiments))
        '''

if __name__ == '__main__':
    main()
