from typing import List, Dict, Union
import os, math, random
import numpy as np
import matplotlib.pyplot as plt
from utils.sentiment_detection import read_tokens, load_reviews, split_data
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon


def calculate_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior log probability
    """
    pos = 0
    neg = 0
    for data in training_data:
        if data['sentiment'] == -1:
            neg += 1
        else:
            pos += 1
    
    return {1: math.log(pos/len(training_data)), -1: math.log(neg/len(training_data))}


def calculate_unsmoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the unsmoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    pos = {}
    neg = {}
    vocab = set()
    for data in training_data:
        for token in data['text']:
            if token not in vocab:
                vocab.add(token)
            
            if data['sentiment'] == 1:
                if token in pos.keys():
                    pos[token] += 1
                else:
                    pos[token] = 1
            else:
                if token in neg.keys():
                    neg[token] += 1
                else:
                    neg[token] = 1
    pos_total = sum(pos.values())
    neg_total = sum(neg.values())

    for pos_token in pos.keys():
        pos[pos_token] = math.log(pos[pos_token] / pos_total)
    for neg_token in neg.keys():
        neg[neg_token] = math.log(neg[neg_token] / neg_total)
    
    pos['unknown'] = math.log(1 / len(vocab))
    neg['unknown'] = math.log(1 / len(vocab))  
    return {1: pos, -1: neg}


def calculate_smoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment. Use the smoothing
    technique described in the instructions (Laplace smoothing).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: Dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    pos = {}
    neg = {}
    vocab = set()
    for data in training_data:
        for token in data['text']:
            if token not in vocab:
                vocab.add(token)

            if data['sentiment'] == 1:
                if token in pos.keys():
                    pos[token] += 1
                else:
                    pos[token] = 1
            else:
                if token in neg.keys():
                    neg[token] += 1
                else:
                    neg[token] = 1
            
    pos_total = sum(pos.values()) + len(vocab)
    neg_total = sum(neg.values()) + len(vocab)
    for pos_token in pos.keys():
        pos[pos_token] = math.log((pos[pos_token] + 1)/ pos_total)
    
    for neg_token in neg.keys():
        neg[neg_token] = math.log((neg[neg_token] + 1)/ neg_total)
    pos['unknown'] = math.log(1 / len(vocab))
    neg['unknown'] = math.log(1 / len(vocab))
    return {1: pos, -1: neg}


def predict_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                          class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior log probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    pos_prob = class_log_probabilities[1]
    neg_prob = class_log_probabilities[-1]
    pos_dict = log_probabilities[1]
    neg_dict = log_probabilities[-1]
    for token in review:
        if token in pos_dict.keys():
            pos_prob += pos_dict[token]
        else:
            pos_prob += pos_dict['unknown']

        if token in neg_dict.keys():
            neg_prob += neg_dict[token]
        else:
            neg_prob += neg_dict['unknown']
    
    if pos_prob >= neg_prob:
        return 1
    else:
        return -1

def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)
    train_tokenized_data = [{'text': read_tokens(x['filename']), 'sentiment': x['sentiment']} for x in training_data]
    dev_tokenized_data = [read_tokens(x['filename']) for x in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment(review, lexicon)
        preds_simple.append(pred)

    acc_simple = accuracy(preds_simple, validation_sentiments)
    print(f"Your accuracy using simple classifier: {acc_simple}")

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

    
    training_size = np.linspace(0.5, 1, 6)
    epochs = 20
    accuracy_size = np.zeros(shape=(10, 6))
    for epoch in range(epochs):
        random.shuffle(train_tokenized_data)
        for i, size in enumerate(training_size):
            token_number = round(size * len(train_tokenized_data))
            training_vdata = train_tokenized_data[:token_number]
            class_vpriors = calculate_class_log_probabilities(training_vdata)
            smoothed_log_vprobabilities = calculate_smoothed_log_probabilities(training_vdata)
            preds_vsmoothed = []
            for review in dev_tokenized_data:
                pred = predict_sentiment_nbc(review, smoothed_log_vprobabilities, class_vpriors)
                preds_vsmoothed.append(pred)
            acc_vsmoothed = accuracy(preds_vsmoothed, validation_sentiments)
            accuracy_size[epoch, i] = acc_vsmoothed
    
    plt.figure(figsize=(6, 6))
    plt.plot(training_size, accuracy_size.mean(axis=0), linestyle='-', marker='o', color='b', markersize=8)
    for i, (x, y) in enumerate(zip(training_size, accuracy_size.mean(axis=0))):
        plt.text(x, y + 0.0009, f'({x}, {y:.2f})', fontsize=8, ha='left', va='center')
    plt.grid(True)
    plt.xlabel('Training Dataset Size')
    plt.ylabel('Accuracy')
    plt.title("Accuracy with Training Dataset Size")
    plt.savefig("dataset_size.png")
    plt.show()

if __name__ == '__main__':
    main()
