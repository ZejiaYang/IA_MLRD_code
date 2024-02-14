from typing import List, Dict, Union 
import os, math, random
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/yangzejia/Desktop/PartIA/mlrd/Tick3')
from utils.sentiment_detection import read_tokens, load_reviews, split_data
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon


def calculate_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c).
    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1 or 0 (for neutral), for positive and negative sentiments.
    @return: dictionary from sentiment to prior log probability
    """
    labels = [0 for _ in range(3)]
    sents = [1, -1, 0]
    for data in training_data:
        for i in range(3):
            if data['sentiment'] == sents[i]:
                labels[i] += 1
    logs = [math.log(label / len(training_data)) for label in labels]
    return {key:value for key, value in zip(sents,logs)}


def calculate_unsmoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the unsmoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    labels = [{} for _ in range(3)]
    sents = [1, -1, 0]
    vocab = set()
    for data in training_data:
        for token in data['text']:
            if token not in vocab:
                vocab.add(token)
            for i in range(3):
                if data['sentiment'] == sents[i]:
                    if token in labels[i].keys():
                        labels[i][token] += 1
                    else:
                        labels[i][token] = 1

    totals = [sum(label.values()) for label in labels]
    for i in range(3):
        for token in labels[i].keys():
            labels[i][token] = math.log(labels[i][token] / totals[i])
        labels[i]['unknown!'] = math.log(1 / len(vocab))

    return {key:value for key, value in zip(sents,labels)}


def calculate_smoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]], smooth:int = 1) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment. Use the smoothing
    technique described in the instructions (Laplace smoothing).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: Dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    labels = [{} for _ in range(3)]
    sents = [1, -1, 0]
    vocab = set()
    for data in training_data:
        for token in data['text']:
            if token not in vocab:
                vocab.add(token)
            for i in range(3):
                if data['sentiment'] == sents[i]:
                    if token in labels[i].keys():
                        labels[i][token] += 1
                    else:
                        labels[i][token] = 1

    totals = [sum(label.values()) + smooth * len(vocab) for label in labels]
    for i in range(3):
        for token in labels[i].keys():
            labels[i][token] = math.log((labels[i][token] + smooth) / totals[i])
        labels[i]['unknown!'] = math.log(1 / len(vocab))
    
    return {key:value for key, value in zip(sents,labels)}

def calculate_binary_smooth_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]], smooth:int = 1) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment. Use the smoothing
    technique described in the instructions (Laplace smoothing).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: Dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    labels = [{} for _ in range(3)]
    sents = [1, -1, 0]
    vocab = set()
    for data in training_data:
        data_word = set()
        for token in data['text']:
            if token not in vocab:
                vocab.add(token)
            if token in data_word:
                continue
            else:
                data_word.add(token)
        for token in data['text']:
            if token not in vocab:
                vocab.add(token)
            for i in range(3):
                if data['sentiment'] == sents[i]:
                    if token in labels[i].keys():
                        labels[i][token] += 1
                    else:
                        labels[i][token] = 1

    totals = [sum(label.values()) + smooth * len(vocab) for label in labels]
    for i in range(3):
        for token in labels[i].keys():
            labels[i][token] = math.log((labels[i][token] + smooth) / totals[i])
        labels[i]['unknown!'] = math.log(1 / len(vocab))
    
    return {key:value for key, value in zip(sents,labels)}


def predict_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                          class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior log probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    sents = [1, -1, 0]
    probs = [class_log_probabilities[sent] for sent in sents]
    dicts = [log_probabilities[sent] for sent in sents]
    #review_word = set()
    for token in review:
        # if token in review_word:
        #     continue
        # review_word.add(token)
        for i in range(3):
            if token in dicts[i].keys():
                probs[i] += dicts[i][token]
            else:
                probs[i] += dicts[i]['unknown!']
    index = max(range(len(probs)), key=probs.__getitem__)
    return sents[index]

def predict_binary_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                          class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior log probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    sents = [1, -1, 0]
    probs = [class_log_probabilities[sent] for sent in sents]
    dicts = [log_probabilities[sent] for sent in sents]
    review_word = set()
    for token in review:
        if token in review_word:
            continue
        review_word.add(token)
        for i in range(3):
            if token in dicts[i].keys():
                probs[i] += dicts[i][token]
            else:
                probs[i] += dicts[i]['unknown!']
    index = max(range(len(probs)), key=probs.__getitem__)
    return sents[index]

def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join("data", "sentiment_detection", "reviews_nuanced"), include_nuance=True)
    print(len(review_data))
    training_data, validation_data = split_data(review_data, seed=0)
    train_tokenized_data = [{'text': read_tokens(x['filename']), 'sentiment': x['sentiment']} for x in training_data]
    dev_tokenized_data = [read_tokens(x['filename']) for x in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]
    #lexicon = read_lexicon(os.path.join("data", "sentiment_detection", "sentiment_lexicon"))

    # preds_simple = []
    # for review in dev_tokenized_data:
    #     pred = predict_sentiment(review, lexicon)
    #     preds_simple.append(pred)

    # acc_simple = accuracy(preds_simple, validation_sentiments)
    # print(f"Your accuracy using simple classifier: {acc_simple}")

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    unsmoothed_log_probabilities = calculate_unsmoothed_log_probabilities(train_tokenized_data)
    preds_unsmoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, unsmoothed_log_probabilities, class_priors)
        preds_unsmoothed.append(pred)

    acc_unsmoothed = accuracy(preds_unsmoothed, validation_sentiments)
    print(f"Your accuracy using unsmoothed probabilities: {acc_unsmoothed}")

    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)
    preds_smoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_smoothed.append(pred)

    acc_smoothed = accuracy(preds_smoothed, validation_sentiments)
    print(f"Your accuracy using smoothed probabilities: {acc_smoothed}")

    smoothed_binary_log_probabilities = calculate_binary_smooth_log_probabilities(train_tokenized_data)
    preds_smoothed = []
    for review in dev_tokenized_data:
        pred = predict_binary_sentiment_nbc(review, smoothed_binary_log_probabilities, class_priors)
        preds_smoothed.append(pred)
    acc_smoothed = accuracy(preds_smoothed, validation_sentiments)
    print(f"Your accuracy using binary smoothed probabilities: {acc_smoothed}")



if __name__ == '__main__':
    main()
