import matplotlib.pyplot as plt
import os
import glob
import tarfile
from pathlib import Path

sinhala_start = 3456
vowels_and_const_end = 3527
sinhala_end = 3573

def is_sinhala_letter(letter):
    unicode_val = ord(letter)
    # only Sinhala vowels and constants
    if sinhala_start <= unicode_val <= vowels_and_const_end:
        return True
    
def is_sinhala_word(word):
    # A word has more than one Sinhala letter
    # can have other letters/digits; Ex: 25වන
    letter_count = 0
    for letter in word:
        if is_sinhala_letter(letter):
            letter_count += 1
            if letter_count >= 2:
                return True
    return False

def is_strictly_sinhala_word(word):
    for ch in word:
        unicode_val = ord(ch)
        if not (sinhala_start <= unicode_val <= sinhala_end):
            return False
    return True

def word_length(word):
    length = 0
    for letter in word:
        if is_sinhala_letter(letter):
            length += 1
    return length

def words_in_sentence(sentence):
    count = 0
    words = sentence.split()  # splits on whitespaces
    for word in words:
        if is_sinhala_word(word):
            count += 1
    return count

def init_stat(corpus: list, words_dict: dict):
    # corpus is a list of sentences
    # words_dict key->word | value->number of occurrences
    for sentence in corpus:
        words = sentence.split()
        for word in words:
            if is_sinhala_word(word):
                if word in words_dict:
                    words_dict[word] += 1
                else:
                    words_dict[word] = 1

def construct_dataset(dataset_path):
    if not os.path.exists("datasets/stat"):
        os.makedirs("datasets/stat")
    dataset_zip = tarfile.open(dataset_path)
    dataset_zip.extract("dataset.txt", "datasets/stat")
    dataset_zip.close()


ccnet_dir = Path(
    input(
        'Please download the CCNet corpus from https://github.com/facebookresearch/cc_net and enter the path to the downloaded data: '
    )
)

all_words_dict = dict()
reverse_dict = dict()
len_dict = dict()  # key-> length; val-> number of words having that length
sentence_len_dict = dict()  # key-> sentence; val->length of sentence
sentence_len_dict_reverse = dict()  # key->legth of a sentence; val-> number of sentences with that length

with open("datasets/stat/dataset.txt") as dataset:
    sentences = dataset.readlines()
    init_stat(sentences, all_words_dict)

    # sentence length analysis
    for sen in sentences:
        sentence_len_dict[sen] = words_in_sentence(sen)
    for val in sentence_len_dict.values():
        if val in sentence_len_dict_reverse.keys():
            sentence_len_dict_reverse[val] += 1
        else:
            sentence_len_dict_reverse[val] = 1
    plt_sent_len = plt.figure("Words per sentence")
    wps_keys = sorted(sentence_len_dict_reverse.keys())
    wps_values = []
    for i in wps_keys:
        wps_values.append(sentence_len_dict_reverse[i])
    plt.bar(wps_keys, wps_values, align='center')
    plt.xticks(wps_keys)
    plt.xlabel("length")
    plt.ylabel("number of sentences with the length")

    # word length analysis
    words_list = all_words_dict.keys()
    for wrd in words_list:
        ln = word_length(wrd)
        if ln in len_dict:
            len_dict[ln] += 1
        else:
            len_dict[ln] = 1

    plot_len = plt.figure("Syllables per Word")
    len_dict_keys_sorted = sorted(len_dict.keys())
    len_dict_values_sorted = []
    for k in len_dict_keys_sorted:
        len_dict_values_sorted.append(len_dict[k])
    plt.bar(len_dict_keys_sorted, len_dict_values_sorted, align='center')
    plt.xticks(len_dict_keys_sorted)
    plt.xlabel("length")
    plt.ylabel("number of unique words")

    # frequency analysis
    for key in all_words_dict:
        val = all_words_dict[key]
        if val in reverse_dict:
            reverse_dict[val] += 1
        else:
            reverse_dict[val] = 1

    plot_freq = plt.figure("Frequency")
    k_rev = sorted(reverse_dict.keys())
    v_rev = []
    for k in k_rev:
        v_rev.append(reverse_dict[k])
    plt.plot(k_rev, v_rev, '.r')
    plt.xlabel("frequency")
    plt.ylabel("number of words with that frequency")

    unique_num_words = len(all_words_dict.keys())
    total_words = sum(all_words_dict.values())
    print("Total number of words: ", total_words)
    print("Number of unique words: ", unique_num_words)
    print("Number of sentences: ", len(sentences))
    print("Average number of words in a sentence: ", (sum(sentence_len_dict.values()) / len(sentences)))
    print("Max word frequency: ", max(all_words_dict.values()))
    print("Min word frequency: ", min(all_words_dict.values()))
    print("Average word frequency: ", (sum(all_words_dict.values()) / len(all_words_dict)))

    plt.show()
