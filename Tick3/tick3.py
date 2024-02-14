from utils.sentiment_detection import clean_plot, chart_plot, best_fit
from typing import List, Tuple, Callable
import os, math
from tqdm import tqdm


def estimate_zipf(token_frequencies_log: List[Tuple[float, float]], token_frequencies: List[Tuple[int, int]]) \
        -> Callable:
    """
    Use the provided least squares algorithm to estimate a line of best fit in the log-log plot of rank against
    frequency. Weight each word by its frequency to avoid distortion in favour of less common words. Use this to
    create a function which given a rank can output an expected frequency.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: a function estimating a word's frequency from its rank
    """
    [m, c] = best_fit(token_frequencies_log, token_frequencies)

    def predict_freq(rank:int) -> float:
        return math.exp(math.log(rank) * m + c)
    
    return [m, c], predict_freq


def count_token_frequencies(dataset_path: str) -> List[Tuple[str, int]]:
    """
    For each of the words in the dataset, calculate its frequency within the dataset.

    @param dataset_path: a path to a folder with a list of  reviews
    @returns: a list of the frequency for each word in the form [(word, frequency), (word, frequency) ...], sorted by
        frequency in descending order
    """
    freqdict = {}

    with tqdm(total=len(os.listdir(dataset_path))) as pbar:
        for file_name in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                line = f.readlines()
                for l in line:
                    wordline = l.split(" ")
                    for word in wordline:
                        if word not in freqdict:
                            freqdict[word] = 0
                        freqdict[word] += 1
            pbar.update(1)

    reslist = [(k,v) for k,v in freqdict.items()]    
    reslist.sort(key=lambda tup: tup[1], reverse = True) 

    return reslist


def draw_frequency_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the provided chart plotting program to plot the most common 10000 word ranks against their frequencies.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    title = "most common 10000 words ranks against their frequencies"
    xlabel = "word ranks"
    ylabel = "frequency"
    data = [(i + 1, v) for i, (_, v) in enumerate(frequencies[:10000])]

    print(f"Draw Figure 1 {title}")
    chart_plot(data, title, xlabel, ylabel)

def draw_selected_words_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot your 10 selected words' word frequencies (from Task 1) against their
    ranks. c

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    file_path = "zy10words.txt"
    title = "selected words ranks against their frequencies"
    xlabel = "word ranks"
    ylabel = "frequency"
    tenwords = []
    tenfreq = []

    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readlines()
        for l in line:
            wordline = l.split(" ")
            for word in wordline:
                tenwords.append(word[:-1])
            
    token_frequencies = [(i + 1, v) for i, (_, v) in enumerate(frequencies)]
    token_log_frequencies = [(math.log(i + 1), math.log(v)) for i, (_, v) in enumerate(frequencies)]
    _, predict_freq = estimate_zipf(token_log_frequencies, token_frequencies)

    for i, (w, c) in enumerate(frequencies[:10000]):
        if w in tenwords:
            tenfreq.append((i + 1, (c, predict_freq(i + 1))))
    
    with open('output.txt', 'w') as file:
        for (w, (f1, f2)) in tenfreq:
            file.write(f"word: {w:<6} actual frequency: {f1:<6} estimated frequency: {f2:<.3f}\n")

    print(f"Draw Figure 2 {title}")
    chart_plot([(i, c) for (i, (c, _)) in tenfreq], title, xlabel, ylabel)
                           

def draw_zipf(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot the logs of your first 10000 word frequencies against the logs of their
    ranks. Also use your estimation function to plot a line of best fit.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    import matplotlib.pyplot as plt
    title = "most common 10000 words log ranks against their log frequencies"
    xlabel = "log word ranks"
    ylabel = "log frequency"
    token_log_frequencies = [(math.log(i + 1), math.log(v)) for i, (_, v) in enumerate(frequencies)]
    token_frequencies = [(i + 1, v) for i, (_, v) in enumerate(frequencies)]

    print(f"Draw Figure 3 {title}")
    chart_plot(token_log_frequencies[:10000], title, xlabel, ylabel)

    [m, c], predict_freq = estimate_zipf(token_log_frequencies, token_frequencies)
    title_zip = "least-squares algorithm on the log-log scale"

    print(f"Draw Figure 4 {title_zip}")

    with open('output.txt', 'a') as file:
        file.write(f"k: {math.exp(c):{6}.3f},  alpha: {m:{6}.3f}\n")

    chart_plot([(math.log(i), math.log(predict_freq(i))) for i in range(1, 10001)], title_zip, xlabel, ylabel)
    


def compute_type_count(dataset_path: str) -> List[Tuple[int, int]]:
    """
     Go through the words in the dataset; record the number of unique words against the total number of words for total
     numbers of words that are powers of 2 (starting at 2^0, until the end of the data-set)

     @param dataset_path: a path to a folder with a list of  reviews
     @returns: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    datapoint = []
    n = 0
    vocab =set()
    with tqdm(total=len(os.listdir(dataset_path))) as pbar:
        for file_name in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                line = f.readlines()
                for l in line:
                    wordline = l.split(" ")
                    for word in wordline:
                        n += 1
                        if word not in vocab:
                            vocab.add(word)
                        if (n & (n - 1)) == 0:
                            datapoint.append((len(vocab), n))
                pbar.update(1)
    return datapoint




def draw_heap(type_counts: List[Tuple[int, int]]) -> None:
    """
    Use the provided chart plotting program to plot the logs of the number of unique words against the logs of the
    number of total words.

    @param type_counts: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    title =  "logs of types against logs of tokens"
    xlabel = "number of tokens"
    ylabel = "number of types"
    data_point = [(math.log(tokens), math.log(types)) for tokens, types in type_counts]
    print(f"Draw Figure 5 {title}")
    chart_plot(data_point, title, xlabel, ylabel)


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    frequencies = count_token_frequencies(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))

    draw_frequency_ranks(frequencies)
    draw_selected_words_ranks(frequencies)

    clean_plot()
    draw_zipf(frequencies)

    clean_plot()
    tokens = compute_type_count(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    draw_heap(tokens)


if __name__ == '__main__':
    main()
