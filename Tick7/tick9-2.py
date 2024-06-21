from utils.markov_models import load_bio_data
import os
import random
from tick8 import recall_score, precision_score, f1_score
from tick7 import print_matrices
import math

from typing import List, Dict, Tuple

observations = ['B', 'Z'] + ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
states = ['B', 'Z'] + ['i', 'o', 'M'] 

def get_transition_probs_bio(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden feature types using maximum likelihood estimation.

    @param hidden_sequences: A list of feature sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    specific = {}
    countall = {k: 0 for k in states}

    for seq in hidden_sequences:
        for i in range(len(seq)-1):
            specific[(seq[i], seq[i+1])] = specific.get((seq[i], seq[i+1]),0)+1
            countall[seq[i]] += 1

    res = {}
    for i in countall:
        for j in countall:
            if countall.get(i, 0) == 0:
                res[(i, j)] = 0
            else:
                res[(i, j)] = specific.get((i, j), 0)/countall[i]
    return res


def get_emission_probs_bio(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden feature states to visible amino acids, using maximum likelihood estimation.
    @param hidden_sequences: A list of feature sequences
    @param observed_sequences: A list of amino acids sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    specific = {}
    countall = {}
    for hidden, observed in zip(hidden_sequences, observed_sequences):
        for state, out in zip(hidden, observed):
            specific[(state, out)] = specific.get((state, out), 0)+1
            countall[state] = countall.get(state, 0)+1
    
    res = {}
    for state in states:
        for observed in observations:
            if countall.get(state, 0) == 0:
                res[(state, observed)] = 0
            else:
                res[(state, observed)] = specific.get((state, observed), 0)/countall[state]
    print_matrices(res)
    return res


def estimate_hmm_bio(training_data:List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities.

    @param training_data: The biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs_bio(hidden_sequences)
    emission_probs = get_emission_probs_bio(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


def viterbi_bio(observed_sequence, transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model.

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    psi = []
    delta = {'B': 1}
    for o in observed_sequence+['Z']:
        new_delta = {}
        psi.append({})
        for (old, new) in transition_probs:
            if transition_probs[(old, new)] == 0 or emission_probs[(new, o)] == 0 or old not in delta:
                continue
            v = math.log(transition_probs[(old, new)]) + math.log(emission_probs[(new, o)]) + delta[old]
            if new not in new_delta or v > new_delta[new]:
                new_delta[new] = v
                psi[-1][new] = old
        delta = new_delta

    path = ['Z']
    for p in psi[::-1]:
        path.insert(0, p[path[0]])
    return path[1:-1]




def self_training_hmm(training_data: List[Dict[str, List[str]]], dev_data: List[Dict[str, List[str]]],
    unlabeled_data: List[List[str]], num_iterations: int) -> List[Dict[str, float]]:
    """
    The self-learning algorithm for your HMM for a given number of iterations, using a training, development, and unlabeled dataset (no cross-validation to be used here since only very limited computational resources are available.)

    @param training_data: The training set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param dev_data: The development set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param unlabeled_data: Unlabeled sequence data of amino acids, encoded as a list of sequences.
    @num_iterations: The number of iterations of the self_training algorithm, with the initial HMM being the first iteration.
    @return: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    """
    dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev['hidden']] for dev in dev_data]
    scores = []
    extra_training = []
    for i in range(num_iterations+1):
        transition, emission = estimate_hmm_bio(training_data + extra_training)
        extra_training = []
        for d in unlabeled_data:
            extra_training.append({'observed': d, 'hidden': viterbi_bio(d, transition, emission)})
        preds = [viterbi_bio(dev['observed'], transition, emission) for dev in dev_data]
        preds_binarized = [[1 if x=='M' else 0 for x in pred] for pred in preds]
        scores.append({'recall': recall_score(preds_binarized, dev_hidden_sequences_binarized), 'precision': precision_score(preds_binarized, dev_hidden_sequences_binarized), 'f1': f1_score(preds_binarized, dev_hidden_sequences_binarized)})
    return scores


def visualize_scores(score_list:List[Dict[str,float]]) -> None:
    """
    Visualize scores of the self-learning algorithm by plotting iteration vs scores.

    @param score_list: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    @return: The most likely single sequence of hidden states
    """
    from utils.sentiment_detection import clean_plot, chart_plot
    recall = [d['recall'] for d in score_list]
    precision = [d['precision'] for d in score_list]
    f1 = [d['f1'] for d in score_list]
    chart_plot([(i, v) for i, v in enumerate(recall)], 'recall', 'iteration', 'score')
    chart_plot([(i, v) for i, v in enumerate(precision)], 'precision', 'iteration', 'score')
    chart_plot([(i, v) for i, v in enumerate(f1)], 'f1', 'iteration', 'score')



def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    bio_data = load_bio_data(os.path.join('data', 'markov_models', 'bio_dataset.txt'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    bio_data_shuffled = random.sample(bio_data, len(bio_data))
    dev_size = int(len(bio_data_shuffled) / 10)
    train = bio_data_shuffled[dev_size:]
    dev = bio_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm_bio(train)

    for sample in dev_observed_sequences:
        prediction = viterbi_bio(sample, transition_probs, emission_probs)
        predictions.append(prediction)
    predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    unlabeled_data = []
    with open(os.path.join('data', 'markov_models', 'bio_dataset_unlabeled.txt'), encoding='utf-8') as f:
        content = f.readlines()
        for i in range(0, len(content), 2):
            unlabeled_data.append(list(content[i].strip())[1:])

    scores_each_iteration = self_training_hmm(train, dev, unlabeled_data, 5)

    visualize_scores(scores_each_iteration)



if __name__ == '__main__':
    main()
