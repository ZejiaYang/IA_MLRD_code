import os
from typing import List, Dict, Tuple
import math
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon
from exercises.tick2 import calculate_class_log_probabilities, calculate_smoothed_log_probabilities, predict_sentiment_nbc
from utils.sentiment_detection import read_tokens, load_reviews, split_data


def read_lexicon_magnitude(filename: str) -> Dict[str, Tuple[int, str]]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to a tuple of sentiment (1, -1) and magnitude ('strong', 'weak').
    """
    lexicon = {}
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if line:
            pairs = line.split()
            data = dict(pair.split('=') for pair in pairs)
            value = []
            if data['polarity'] == 'negative':
                value.append(-1)
            else:
                value.append(1)
            value.append(data['intensity'])
            lexicon[data['word']] = tuple(value)
            
    return lexicon 


def predict_sentiment_magnitude(review: List[str], lexicon: Dict[str, Tuple[int, str]]) -> int:
    """
    Modify the simple classifier from Tick1 to include the information about the magnitude of a sentiment. Given a list
    of tokens from a tokenized review and a lexicon containing both sentiment and magnitude of a word, determine whether
    the sentiment of each review in the test set is positive or negative based on whether there are more positive or
    negative words. A word with a strong intensity should be weighted *four* times as high for the evaluator.

    @param review: list of tokens from tokenized review
    @param lexicon: dictionary from word to a tuple of sentiment (1, -1) and magnitude ('strong', 'weak').
    @return: calculated sentiment for each review (+1 or -1 for positive or negative sentiments respectively).
    """
    sentiment = 0
    magnitude_factor = 4
    for token in review:
        if token in lexicon.keys():
            token_score = lexicon[token][0] 
            if lexicon[token][1] == 'strong':
                token_score *= magnitude_factor
            sentiment += token_score
    if sentiment >= 0:
        return 1
    else:
        return -1


def sign_test(actual_sentiments: List[int], classification_a: List[int], classification_b: List[int]) -> float:
    """
    Implement the two-sided sign test algorithm to determine if one classifier is significantly better or worse than
    another. The sign for a result should be determined by which classifier is more correct and the ceiling of the least
    common sign total should be used to calculate the probability.

    @param actual_sentiments: list of correct sentiment for each review
    @param classification_a: list of sentiment prediction from classifier A
    @param classification_b: list of sentiment prediction from classifier B
    @return: p-value of the two-sided sign test.
    """
    null = 0
    plus = 0 # A is greater than B
    minus = 0 # A is worse than B
    for actual, class_a, class_b in zip(actual_sentiments, classification_a, classification_b):
        if (class_a == actual and class_b != actual):
            plus += 1
        elif (class_a != actual and class_b == actual):
            minus += 1
        else:
            null += 1
    n = 2 * math.ceil(null / 2) + plus + minus
    k = math.ceil(null / 2) + min(plus, minus)
    d = math.pow(2, n)
    sum = 1
    t = 1
    for i in range(1, k + 1):
        t = t * (n - i + 1) / i
        sum += t
    p_value = 2 * sum / d
    return p_value


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    training_data, validation_data = split_data(review_data, seed=0)

    train_tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in training_data]
    dev_tokenized_data = [read_tokens(fn['filename']) for fn in review_data]
    validation_sentiments = [x['sentiment'] for x in review_data]

    lexicon_magnitude = read_lexicon_magnitude(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))
    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_magnitude = []
    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_magnitude(review, lexicon_magnitude)
        preds_magnitude.append(pred)
        pred_simple = predict_sentiment(review, lexicon)
        preds_simple.append(pred_simple)

    acc_magnitude = accuracy(preds_magnitude, validation_sentiments)
    acc_simple = accuracy(preds_simple, validation_sentiments)


    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)

    preds_nb = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_nb.append(pred)

    acc_nb = accuracy(preds_nb, validation_sentiments)
    
    print(f"Your accuracy using simple classifier: {acc_simple}")
    print(f"Your accuracy using magnitude classifier: {acc_magnitude}")
    print(f"Your accuracy using Naive Bayes classifier: {acc_nb}\n")

    classifiers = [('simple', preds_simple),('magnitude', preds_magnitude), ('naive bayes', preds_nb)]

    for s1, c1 in classifiers:
        for s2, c2 in classifiers:
            print(f"a: {s1} b: {s2} p-value: {sign_test(validation_sentiments, c1, c2)}")

    '''Smoothing Factor'''
    for s in range(1, 11):
        smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data, smooth=s)
        preds_nb_i = []
        for review in dev_tokenized_data:
            pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
            preds_nb_i.append(pred)
        ac = accuracy(preds_nb_i, validation_sentiments)
        print(f"smooth-factor: {s}, acc: {ac}, p-value: {sign_test(validation_sentiments, preds_nb, preds_nb_i)}")
        
if __name__ == '__main__':
    main()
