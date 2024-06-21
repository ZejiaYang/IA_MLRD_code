from utils.markov_models import load_bio_data
import os
import random
from exercises.tick8 import recall_score, precision_score, f1_score
import math
from typing import List, Dict, Tuple


def get_transition_probs_bio(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden feature types using maximum likelihood estimation.

    @param hidden_sequences: A list of feature sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    trans_table = dict()
    trans_from = dict()
    vocab = []
    for sequence in hidden_sequences:
        for i in range(len(sequence) - 1):
            if i == len(sequence) - 2:
                if sequence[i + 1] not in vocab:
                    vocab.append(sequence[i + 1])
                if sequence[i + 1] not in trans_from.keys():
                    trans_from[sequence[i + 1]] = 0

            if sequence[i] not in vocab:
                vocab.append(sequence[i])
                trans_from[sequence[i]] = 1
            else:
                trans_from[sequence[i]] += 1

            if (sequence[i], sequence[i + 1]) not in trans_table.keys():
                trans_table[(sequence[i], sequence[i + 1])] = 1
            else:
                trans_table[(sequence[i], sequence[i + 1])] += 1   

    for i in range(len(vocab)):
        for j in range(len(vocab)):
            if (vocab[i], vocab[j]) not in trans_table.keys():
                trans_table[(vocab[i], vocab[j])] = 0
    trans_table = {key: value/trans_from[key[0]] if trans_from[key[0]] != 0 else 0 for key, value in trans_table.items()}
    return trans_table


def get_emission_probs_bio(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden feature states to visible amino acids, using maximum likelihood estimation.
    @param hidden_sequences: A list of feature sequences
    @param observed_sequences: A list of amino acids sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    emission_table = dict()
    trans_from = dict()
    vocab_h = []
    vocab_d = []
    for hs, os in zip(hidden_sequences, observed_sequences):
        for h, o in zip(hs, os):
            if h not in vocab_h:
                vocab_h.append(h)
                trans_from[h] = 1
            else:
                trans_from[h] += 1
            if o not in vocab_d:
                vocab_d.append(o)          
            if (h, o) not in  emission_table.keys():
                emission_table[(h, o)] = 1
            else:
                emission_table[(h, o)] += 1

    for i in range(len(vocab_h)):
        for j in range(len(vocab_d)):
            if (vocab_h[i], vocab_d[j]) not in emission_table.keys():
                emission_table[(vocab_h[i], vocab_d[j])] = 0
    emission_table = {key: value/trans_from[key[0]] if trans_from[key[0]] != 0 else 0 for key, value in  emission_table.items()}
    return emission_table


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
    states = set([pair[0] for pair in transition_probs.keys()])
    path_prob = [{s: float('-inf') if s != 'B' else 0 for s in states}]
    pre_state = [{s: None for s in states}]
    # Here include Z in the observed_sequence
    for sequece in observed_sequence + ['Z']:
        path_t = {}
        prev_s = {}
        last_prob = path_prob[-1]
        for state in states:
            path_t_prob = float('-inf')
            prev_t_state = None
            if emission_probs[(state, sequece)] != 0:
                for prev in states:
                    if transition_probs[(prev, state)] == 0:
                        continue
                    else:
                        if path_t_prob < math.log(transition_probs[(prev, state)]) + last_prob[prev]:
                            path_t_prob = math.log(transition_probs[(prev, state)]) + last_prob[prev]
                            prev_t_state = prev
                if prev_t_state != None: 
                    path_t_prob += math.log(emission_probs[(state, sequece)])
            path_t[state] = path_t_prob
            prev_s[state] = prev_t_state
        path_prob.append(path_t)
        pre_state.append(prev_s)
    pred_state = []
    last_s =  max(path_prob[-1], key=lambda k: path_prob[-1][k])
    for i in range(len(path_prob) - 1, 0, -1):
        pred_state.append(last_s)
        last_s = pre_state[i][last_s]
    pred_state.reverse()
    return pred_state[:-1]




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
    dev_observed_sequences = [x['observed'] for x in dev_data]
    dev_hidden_sequences = [x['hidden'] for x in dev_data]
    dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]
    metrices = []
    label_predictions = []
    for _ in range(num_iterations + 1):
        transition_probs, emission_probs = estimate_hmm_bio(training_data + label_predictions)
        predictions = [viterbi_bio(sample, transition_probs, emission_probs) for sample in dev_observed_sequences]
        predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]

        p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
        r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
        f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)
        metrices.append({"recall":r, "precision":p, "f1": f1})

        label_predictions = [{'observed': sample, 'hidden': viterbi_bio(sample, transition_probs, emission_probs)} \
                             for sample in unlabeled_data]
        # Here everytime relabelled them
    return metrices

def visualize_scores(score_list:List[Dict[str,float]]) -> None:
    """
    Visualize scores of the self-learning algorithm by plotting iteration vs scores.

    @param score_list: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    @return: The most likely single sequence of hidden states
    """
    from utils.sentiment_detection import clean_plot, chart_plot
    metrices = ['recall', 'precision', 'f1']
    for metrice in metrices:
        clean_plot()
        p = [(i, x[f'{metrice}']) for i, x in enumerate(score_list)]
        chart_plot(p, f"{metrice} with iterations", "iteration", f"{metrice}")
    
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
