from genericpath import exists
import matplotlib.pyplot as plt
import os
import glob
import tarfile
from pathlib import Path
import fasttext
import numpy as np
import codecs
import argparse
import os.path
from os import path

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


# ccnet_dir = Path(
#     input(  
#         'Please download the CCNet corpus from https://github.com/facebookresearch/cc_net and enter the path to the downloaded data: '
#     )
# )

all_words_dict = dict()
reverse_dict = dict()
len_dict = dict()  # key-> length; val-> number of words having that length
sentence_len_dict = dict()  # key-> sentence; val->length of sentence
sentence_len_dict_reverse = dict()  # key->legth of a sentence; val-> number of sentences with that length
sentence_lang_dict = dict()


parser = argparse.ArgumentParser()
parser.add_argument("file_path", help="File path of the data file")
args = parser.parse_args()
file_path = args.file_path

if (not path.exists(file_path)):
    print("File %s doesn't exist. Verify file path." % file_path)
else:
    with open(file_path) as dataset:
        sentences = dataset.readlines()
        init_stat(sentences, all_words_dict)
        if not os.path.exists("resources/diagrams"):
            os.makedirs("resources/diagrams")
        if not os.path.exists("resources/reports"):
            os.makedirs("resources/reports")
        report_file = codecs.open("resources/reports/report.txt", "w+", "utf-8")

        # sentence length analysis
        for sen in sentences:
            sentence_len_dict[sen] = words_in_sentence(sen)
        for val in sentence_len_dict.values():
            if val > 400:
                continue
            if val in sentence_len_dict_reverse.keys():
                sentence_len_dict_reverse[val] += 1
            else:
                sentence_len_dict_reverse[val] = 1
        plt_sent_len = plt.figure("Words per sentence")
        wps_keys = sorted(sentence_len_dict_reverse.keys())
        wps_values = []
        for i in wps_keys:
            wps_values.append(sentence_len_dict_reverse[i])
        #wps_keys_gap = sorted(sentence_len_dict_reverse.keys(), key=lambda k: k%4 == 0)
        plt.bar(wps_keys, wps_values, align='center')
        plt.xticks(wps_keys)
        plt.xticks(rotation=90, fontsize=4)
        
        plt.xlabel("length")
        plt.ylabel("number of sentences with the length")
        plt.savefig("resources/diagrams/sentence_length.eps", format="eps", dpi=1200)
        plt.savefig("resources/diagrams/sentence_length.svg", format="svg", dpi=1200)
        report_file.write("-------------------Sentence report-------------------\n")
        report_file.write("\n")
        max_sorted_sentence_len_dict = dict(sorted(sentence_len_dict.items(), key=lambda kv: kv[1], reverse=True)[:100])
        sentence_len_dict_zero_removed = {key:val for key, val in sentence_len_dict.items() if val != 0}
        min_sorted_sentence_len_dict = dict(sorted(sentence_len_dict_zero_removed.items(), key=lambda kv: kv[1])[:100])
        report_file.write("Maximum 100 sentences\n")
        for key, value in max_sorted_sentence_len_dict.items():
            report_file.write("{key}\n".format(key=key, value=value))
        report_file.write("\n")
        report_file.write("Minimum 100 sentences\n")
        for key, value in min_sorted_sentence_len_dict.items():
            report_file.write("{key}\n".format(key=key, value=value))
        report_file.write("\n")

        # word length analysis
        words_list = all_words_dict.keys()
        for wrd in words_list:
            ln = word_length(wrd)
            if ln > 150:
                continue
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
        plt.xticks(rotation=90, fontsize=4)
        plt.xlabel("length")
        plt.ylabel("number of unique words")
        plt.savefig("resources/diagrams/word_length.eps", format="eps", dpi=1200)
        plt.savefig("resources/diagrams/word_length.svg", format="svg", dpi=1200)
        report_file.write("-------------------Word report-------------------\n")
        report_file.write("\n")
        max_sorted_word_len_dict = dict(sorted(all_words_dict.items(), key=lambda kv: kv[1], reverse=True)[:100])
        word_len_dict_zero_removed = {key:val for key, val in all_words_dict.items() if val != 0}
        min_sorted_word_len_dict = dict(sorted(word_len_dict_zero_removed.items(), key=lambda kv: kv[1])[:100])
        report_file.write("Maximum 100 words\n")
        for key, value in max_sorted_word_len_dict.items():
            report_file.write("{key} - {value}\n".format(key=key, value=value))
        report_file.write("\n")
        report_file.write("Minimum 100 words\n")
        for key, value in min_sorted_word_len_dict.items():
            report_file.write("{key} - {value}\n".format(key=key, value=value))
        report_file.write("\n")


        # frequency analysis
        for key in all_words_dict:
            val = all_words_dict[key]
            if val > 150:
                continue
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
        plt.savefig("resources/diagrams/frequency.eps", format="eps", dpi=1200)
        plt.savefig("resources/diagrams/frequency.svg", format="svg", dpi=1200)

        unique_num_words = len(all_words_dict.keys())
        total_words = sum(all_words_dict.values())
        report_file.write("-------------------Summary-------------------\n")
        report_file.write("\n")
        print("Total number of words: ", total_words)
        print("Number of unique words: ", unique_num_words)
        print("Number of sentences: ", len(sentences))
        print("Average number of words in a sentence: ", (sum(sentence_len_dict.values()) / len(sentences)))
        print("Max word frequency: ", max(all_words_dict.values()))
        print("Min word frequency: ", min(all_words_dict.values()))
        print("Average word frequency: ", (sum(all_words_dict.values()) / len(all_words_dict)))
        report_file.write("Total number of words: {total}\n".format(total=total_words))
        report_file.write("Number of unique words: {unique}\n".format(unique=unique_num_words))
        report_file.write("Number of sentences: {sentences}\n".format(sentences=len(sentences)))
        report_file.write("Average number of words in a sentence: {avg}\n".format(avg=(sum(sentence_len_dict.values()) / len(sentences))))
        report_file.write("Max word frequency: {max}\n".format(max=max(all_words_dict.values())))
        report_file.write("Min word frequency: {min}\n".format(min=min(all_words_dict.values())))
        report_file.write("Average word frequency:{avg_word}\n".format(avg_word=(sum(all_words_dict.values()) / len(all_words_dict))))

        plt.show()
