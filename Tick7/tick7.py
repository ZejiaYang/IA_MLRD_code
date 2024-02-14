from utils.markov_models import load_dice_data, print_matrices
import os
from typing import List, Dict, Tuple


def get_transition_probs(hidden_sequences: List[List[str]]) \
    -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden dice types using maximum likelihood estimation. 
    Counts the number of times each state sequence appears and divides it by the count of all transitions going from that state. 
    The table must include proability values for all state-state pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
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
                    trans_from[sequence[i + 1]] = 1
                else:
                    trans_from[sequence[i + 1]] += 1
            
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
    trans_table = {key: value/trans_from[key[0]] for key, value in trans_table.items()}
    return trans_table

def get_emission_probs(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) \
    -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden dice states to observed dice rolls, using maximum likelihood estimation. 
    Counts the number of times each dice roll appears for the given state (fair or loaded) and divides it by the count of that state. 
    The table must include proability values for all state-observation pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @param observed_sequences: A list of dice roll sequences
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
    emission_table = {key: value/trans_from[key[0]] for key, value in  emission_table.items()}
    return emission_table


def estimate_hmm(training_data: List[Dict[str, List[str]]]) \
    -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities. 
    We use 'B' for the start state and 'Z' for the end state, both for emissions and transitions.

    @param training_data: The dice roll sequence data (visible dice rolls and hidden dice types), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs(hidden_sequences)
    emission_probs = get_emission_probs(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))
    transition_probs, emission_probs = estimate_hmm(dice_data)
    print(f"The transition probabilities of the HMM:")
    print_matrices(transition_probs)
    print(f"The emission probabilities of the HMM:")
    print_matrices(emission_probs)

if __name__ == '__main__':
    main()
