import os
from typing import List, Dict, Union
import math
from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table

from exercises.tick5 import generate_random_cross_folds, cross_validation_accuracy


def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    labels = [0 for _ in range(3)]
    sents = [1, -1, 0]
    for data in training_data:
        for i in range(3):
            if data['sentiment'] == sents[i]:
                labels[i] += 1
    logs = [math.log(label / len(training_data)) for label in labels]
    return {key:value for key, value in zip(sents,logs)}


def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]], smooth=1) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
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

    totals = [sum(label.values()) + smooth * len(vocab) for label in labels]
    for i in range(3):
        for token in labels[i].keys():
            labels[i][token] = math.log((labels[i][token] + smooth) / totals[i])
        labels[i]['unknown!'] = math.log(1 / len(vocab))
    
    return {key:value for key, value in zip(sents,labels)}



def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    equal = [1 if p == t else 0 for p, t in zip(pred, true)]
    return sum(equal)/len(pred)


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                                  class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 0, 1] for the given review
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


def calculate_kappa(agreement_table: Dict[int, Dict[int,int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    sents = [1, -1, 0]
    pe_sum = 0
    sents_num = {key:0 for key in sents}
    for _, review in agreement_table.items():
        for pred, num in review.items():
            sents_num[pred] += num
            pe_sum += math.pow(num, 2)
    total = sum(sents_num.values())
    pe = sum(math.pow(num/total,2) for num in sents_num.values())
    n = len(agreement_table.keys()) #number of document
    N = total / n #total number of annotator

    pa = (pe_sum - N*n)/ (n * N * (N - 1))
    return (pa - pe) / (1 - pe)


def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int,int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    agree_table = dict()
    for review in review_predictions:
        for _, (i, pred) in enumerate(review.items()):
            if i not in agree_table.keys():
                agree_table[i] = {-1:0, 1:0, 0:0}
            agree_table[i][pred] += 1
   #print(agree_table.keys())
    #print(agree_table.values())
    #print(agree_table)
    return agree_table
    


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        test = split_training_data[i]
        train_unflattened = split_training_data[:i] + split_training_data[i+1:]
        train = [item for sublist in train_unflattened for item in sublist]

        dev_tokens = [x['text'] for x in test]
        dev_sentiments = [x['sentiment'] for x in test]

        class_priors = nuanced_class_log_probabilities(train)
        nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
        preds_nuanced = []
        for review in dev_tokens:
            pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
            preds_nuanced.append(pred)
        acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
        accuracies.append(acc_nuanced)

    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'Manual Sentiment Assignment-91-records-20240202_1547-comma_separated.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    review_predictions_four_years = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2023.csv'))
    agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    print('Agreement table for the years 2019 to 2023.')
    print_agreement_table(agreement_table_four_years)

    fleiss_kappa = calculate_kappa(agreement_table_four_years)

    print(f"The Fliess(?) kappa score for the review predictions from 2019 to 2023 is {fleiss_kappa}.")



if __name__ == '__main__':
    main()
