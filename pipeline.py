from typing import List, Text
from tokenizer import SinhalaTokenizer
import codecs
import os
import glob

tokenizer = SinhalaTokenizer()


def tokenize_line(line: Text) -> List[Text]:
    """
    tokenize_line takes a line as the input, iterates over the sentences 
    within the line, tokenize the sentences and returns the tokenized
    sentences as a list.  
    """
    sentences = tokenizer.split_sentences(line)
    tokenized_sentences = []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        tokenized_sentence = " ".join(tokens) + "\n"
        tokenized_sentences.append(tokenized_sentence)
    return tokenized_sentences


def tokenize_directory(directory="datasets/raw"):
    """
    tokenize_directory is the start of the pipeline. It will take an
    input directory with text files, tokenize every sentence and save
    the tokenized sentences in a temporary text file. 
    """
    initialize_directory_structure()
    temp_file = codecs.open("datasets/temp/temp.txt", "w+", "utf-8")
    for source_file in glob.glob(os.path.join(directory, '*.txt')):
        with open(source_file) as file:
            for line in file:
                tokenized_sentences = tokenize_line(line)
                for tokenized_sentence in tokenized_sentences:
                    if len(tokenized_sentence) > 20:
                        temp_file.write(tokenized_sentence)
    temp_file.close()


def write_to_shards():
    """
    write_to_shards reads the tokenized sentences in the tempfile and
    write those sentences into small text files specified by a limit.
    """
    lines_per_file = 100000
    smallfile = None
    with open("datasets/temp/temp.txt") as tempfile:
        for lineno, line in enumerate(tempfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = 'datasets/tokenized/tokenized_shard_{}.txt'.format(
                    lineno + lines_per_file)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()


def initialize_directory_structure():
    """
    Helper method to initiate directory structure.
    """
    if not os.path.exists("datasets/temp"):
        os.makedirs("datasets/temp")
    if not os.path.exists("datasets/tokenized"):
        os.makedirs("datasets/tokenized")


# Pipeline steps
tokenize_directory()
write_to_shards()
