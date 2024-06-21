from utils.markov_models import load_dice_data
import os
from tick7 import estimate_hmm
import random
import math
from typing import List, Dict, Tuple


def viterbi(observed_sequence: List[str], transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model. Use the same symbols for the start and end observations as in tick 7 ('B' for the start observation and 'Z' for the end observation).

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    # states = set([pair[0] for pair in transition_probs.keys()])
    # # path_prob = [{s: float('-inf') if emission_probs[(s, observed_sequence[0])] == 0 else math.log(emission_probs[(s, observed_sequence[0])])
    # #                for s in states}]
    # path_prob = [{s: float('-inf') if s != 'B' else 0 for s in states}]
    # pre_state = [{s: None for s in states}]
    # for sequece in observed_sequence:
    #     path_t = {}
    #     prev_s = {}
    #     last_prob = path_prob[-1]
    #     for state in states:
    #         path_t_prob = float('-inf')
    #         prev_t_state = None
    #         if emission_probs[(state, sequece)] != 0:
    #             for prev in states:
    #                 if transition_probs[(prev, state)] == 0:
    #                     continue
    #                 else:
    #                     if path_t_prob < math.log(transition_probs[(prev, state)]) + last_prob[prev]:
    #                         path_t_prob = math.log(transition_probs[(prev, state)]) + last_prob[prev]
    #                         prev_t_state = prev
    #             if prev_t_state != None: # Find one
    #                 path_t_prob += math.log(emission_probs[(state, sequece)])
    #         path_t[state] = path_t_prob
    #         prev_s[state] = prev_t_state
    #     path_prob.append(path_t)
    #     pre_state.append(prev_s)
    # pred_state = []
    # last_s =  max(path_prob[-1], key=lambda k: path_prob[-1][k])
    # for i in range(len(path_prob) - 1, 0, -1):
    #     pred_state.append(last_s)
    #     last_s = pre_state[i][last_s]
    # pred_state.reverse()
    # return pred_state
    states = set([pair[0] for pair in transition_probs.keys()])
    # path_prob = [{s: float('-inf') if emission_probs[(s, observed_sequence[0])] == 0 else math.log(emission_probs[(s, observed_sequence[0])])
    #                for s in states}]
    path_prob = [{s: float('-inf') if s != 'B' else 0 for s in states}]
    pre_state = [{s: None for s in states}]
    for sequece in observed_sequence:
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
    return pred_state

def precision_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the precision of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of predicted weighted states that were actually weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The precision of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    tp_total = 0
    fp_total = 0
    
    for pred_seq, true_seq in zip(pred, true):
        tp = sum(1 for p, t in zip(pred_seq, true_seq) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(pred_seq, true_seq) if p == 1 and t == 0)
        
        tp_total += tp
        fp_total += fp
    
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0
    
    return precision


def recall_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the recall of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of actual weighted states that were predicted weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The recall of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    tp_total = 0
    fn_total = 0
    
    for pred_seq, true_seq in zip(pred, true):
        tp = sum(1 for p, t in zip(pred_seq, true_seq) if p == 1 and t == 1)
        fn = sum(1 for p, t in zip(pred_seq, true_seq) if p == 0 and t == 1)
        
        tp_total += tp
        fn_total += fn
    
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0
    
    return recall



def f1_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the F1 measure of the estimated sequence with respect to the positive class (weighted state), i.e. the harmonic mean of precision and recall.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The F1 measure of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    precision = precision_score(pred, true)
    recall = recall_score(pred, true)
    return 2 * precision * recall / (precision + recall)


def cross_validation_sequence_labeling(data: List[Dict[str, List[str]]]) -> Dict[str, float]:
    """
    Run 10-fold cross-validation for evaluating the HMM's prediction with Viterbi decoding. Calculate precision, recall, and F1 for each fold and return the average over the folds.

    @param data: the sequence data encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'
    @return: a dictionary with keys 'recall', 'precision', and 'f1' and its associated averaged score.
    """
    fold_num = 10
    fold_sample = int(len(data) // 10)
    recall = 0
    precision = 0
    f1 = 0
    for i in range(fold_num):
        if i == fold_num - 1:
            testing_data = data[i * fold_sample : ]
            training_data = data[0 : i * fold_sample]
        else:
            training_data = data[0 : i * fold_sample] + data[(i + 1) * fold_sample: ]
            testing_data = data[i * fold_sample: (i + 1) * fold_sample]

        transition_probs, emission_probs = estimate_hmm(training_data)
        observed_sequence = [x['observed'] for x in testing_data]
        true_sequence = [x['hidden'] for x in testing_data]
        prediction = []
        for observe in observed_sequence:
            pred = viterbi(observe, transition_probs, emission_probs)
            prediction.append(pred)
        prediction = [[int(x == 'W') for x in pred] for pred in prediction]
        truth = [[int(x == 'W') for x in tu] for tu in true_sequence]
        recall += recall_score(prediction, truth)
        precision += precision_score(prediction, truth)
        f1 += f1_score(prediction, truth)

    return  {'recall': recall/fold_num, 'precision':precision/fold_num, 'f1': f1/fold_num}

    


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    dice_data_shuffled = random.sample(dice_data, len(dice_data))
    dev_size = int(len(dice_data) / 10)
    train = dice_data_shuffled[dev_size:]
    dev = dice_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm(train)

    for sample in dev_observed_sequences:
        prediction = viterbi(sample, transition_probs, emission_probs)
        predictions.append(prediction)

    predictions_binarized = [[1 if x == 'W' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    print(f"Evaluating HMM using cross-validation with 10 folds.")

    cv_scores = cross_validation_sequence_labeling(dice_data)

    print(f" Your cv average precision using the HMM: {cv_scores['precision']}")
    print(f" Your cv average recall using the HMM: {cv_scores['recall']}")
    print(f" Your cv average F1 using the HMM: {cv_scores['f1']}")



if __name__ == '__main__':
    main()
