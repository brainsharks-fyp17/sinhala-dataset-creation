import re
from typing import Tuple, Text, Dict, List


Boolean = bool

__all__ = [
    'SinhalaTokenizer'
]


def is_a_sinhala_letter(s: Text) -> Boolean:
    if len(s) != 1:
        return True
    sinhala_lower_bound = 3456
    sinhala_upper_bound = 3583
    cp = ord(s[0])  # first letter of str
    if sinhala_lower_bound <= cp <= sinhala_upper_bound:
        return True
    return False


def contains_sinhala(s: Text) -> Boolean:
    for c in s:
        if is_a_sinhala_letter(c):
            return True
    return False


# noinspection SpellCheckingInspection
class Tokenizer:
    def tokenize(self, sentence: Text) -> List[Text]:
        raise NotImplementedError()


# noinspection SpellCheckingInspection
class SinhalaTokenizer(Tokenizer):
    def __init__(self):
        self.isolate_punctuations_with_spaces = False
        self.punctuation_marks = [
            '.', ',', '\n', ' ', '¸', '‚','෴',
            '"', '/', '-', '|', '\\', '—', '¦',
            '”', '‘', '\'', '“', '’', '´', '´',
            '!', '@', '#', '$', '%', '^', '&', '*', '+', '-', '£', '?', '˜',
            '(', ')', '[', ']', '{', '}',
            ':', ';',
            '\u2013'  # EN - DASH
        ]
        self.invalid_chars = [
            'Ê',
            '\u00a0', '\u2003',  # spaces
            '\ufffd', '\uf020', '\uf073', '\uf06c', '\uf190',  # unknown or invalid unicode chars
            '\u202a', '\u202c', '\u200f'  # direction control chars(for arabic, starting from right etc)
        ]
        self.line_tokenizing_chars = [
            '.', '?', '!', ':', ';', '\u2022'
        ]
        self.punctuations_without_line_tokenizing_chars = [
            ',', '¸', '‚',
            '"', '/', '-', '|', '\\', '—', '¦',
            '”', '‘', '\'', '“', '’', '´', '´',
            '!', '@', '#', '$', '%', '^', '&',
            '*', '+', '-', '£', '?', '˜',
            '(', ')', '[', ']', '{', '}',
            ':', ';',
            '\u2013'
        ]
        self.short_forms = [
            'ඒ.', 'බී.', 'සී.', 'ඩී.', 'ඊ.', 'එෆ්.', 'ජී.', 'එච්.',
            'අයි.', 'ජේ.', 'කේ.', 'එල්.', 'එම්.', 'එන්.', 'ඕ.',
            'පී.', 'කිව්.', 'ආර්.', 'එස්.', 'ටී.', 'ඩබ.', 'ඩබ්ලිව්.',
            'එක්ස්.', 'වයි.', 'ඉසෙඩ්.',
            'පෙ.', 'ව.', 'වී.',
            'රු.',
            'පා.',  # parliment
            '0.', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'
        ]

        self.number_bullets = ['0.', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.',]

        # Do not use `short_form_identifier` at `punctuation_marks`
        self.short_form_identifier = '\u0D80'

        #  init ignoring chars
        self.ignoring_chars = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
            'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            '\u200c', '\u0160', '\u00ad', '\u0088', '\uf086', '\u200b', '\ufeff', 'Á', 'À', '®', '¡', 'ª', 'º', '¤','>','<'
            '¼', '¾', 'Ó', 'ø', '½', 'ˆ', '', '¢', 'ÿ', '·', 'í', 'Ω', '°', '×', 'µ', '', '~', 'ƒ', '', 'ë', 'Î','◞',
            '‰', '»', '«', 'à', '«', '·', '¨', '…', '⋆', '›', '¥', '⋆', '', '˝', '', '', '◊', 'Ł', '', 'ê', 'Õ', 'Ä',
            'á', 'Ñ', 'Í', '', 'Ñ', 'ç', 'Æ', 'ô', 'Ž', '€', '§', 'Æ', '÷', 'é', '¯', 'é', 'æ', 'î', 'ï', 'ä', 'Ô', 'õ','→',
            'È', 'Ý', 'ß', 'õ', '', 'ù', 'å', 'Ø', 'Œ', 'Ô', 'Ü', '', 'Ö', 'Û', 'Ï', 'ñ', 'ý', 'œ', '¹', '', 'É', '¯','❤',
            'Ò', '`','"','´','“','”','*','♦','█','=','','_',''
        ]

        # init word tokenizer
        self.word_tokenizer_delims = '[{}]'.format(
            re.escape(''.join(self.punctuation_marks + self.invalid_chars)))

        # init line tokenizer
        self.line_tokenizer_delims = '[{}]'.format(re.escape(''.join(self.line_tokenizing_chars)))

    def tokenize(self, sentence: Text) -> List[Text]:
        # remove ignoring chars from document
        for ignoring_char in self.ignoring_chars:
            if ignoring_char in sentence:
                sentence = sentence.replace(ignoring_char, '')

        for number_bullet in self.number_bullets:
            if number_bullet in sentence:
                sentence = sentence.replace(number_bullet, '')

        # prevent short forms being splitted into separate tokens
        # Eg: පෙ.ව.
        for short_form in self.short_forms:
            representation = short_form[0:-1] + self.short_form_identifier
            sentence = sentence.replace(short_form, representation)

        parts = re.split(r'({})'.format(self.word_tokenizer_delims), sentence)
        tokens = [token.replace(self.short_form_identifier, '.') for token in parts if len(token.strip()) != 0]

        #remove punctuations
        new_tokens_without_punctionations = []
        for token in tokens:
            if(token not in self.punctuation_marks):
                new_tokens_without_punctionations.append(token)

        return new_tokens_without_punctionations

    def split_sentences(self, doc: Text, return_sinhala_only: Boolean = False) -> List[Text]:
        # remove ignoring chars from document
        for ignoring_char in self.ignoring_chars:
            if ignoring_char in doc:
                doc = doc.replace(ignoring_char, '')

        # stop words being present with a punctuation at start or end of the word
        # Eg: word?     word,
        if self.isolate_punctuations_with_spaces:  # default is set to FALSE
            for punctuation in self.punctuations_without_line_tokenizing_chars:
                doc = doc.replace(punctuation, ' ' + punctuation + ' ')

        # prevent short forms being splitted into sentences
        # Eg: පෙ.ව.
        for short_form in self.short_forms:
            representation = short_form[0:len(short_form) - 1] + self.short_form_identifier
            doc = doc.replace(short_form, representation)

        #remove text between parenthesis.
        parenthesis_text = re.findall(r'\([^()]+\)', doc)
        for text in parenthesis_text:
            if(len(text) < 40):
                doc = re.sub(r'\([^()]+\)', '', doc)


        sentences = []
        # split lines
        parts = re.split(r'{}'.format(self.line_tokenizer_delims), doc)

        for sentence in parts:
            sentence = sentence.replace(self.short_form_identifier, '.')
            sentence = sentence.strip()
            if contains_sinhala(sentence):  # filter empty sentences and non-sinhala sentences
                sentences.append(sentence)
            elif not return_sinhala_only and len(sentence) != 0:
                sentences.append(sentence)
        return sentences
