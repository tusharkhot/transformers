from typing import List

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string


stemmer = PorterStemmer()

stop_words_set = set(stopwords.words('english'))

spacy_nlp = None

parse_cache = {}

WH_WORDS = ["who", "what", "where", "how", "why", "when", "which"]

def tokenize_str(input_str):
    return word_tokenize(input_str)


def stem_tokens(token_arr):
    return [stemmer.stem(token) for token in token_arr]


def filter_stop_tokens(token_arr):
    return [token for token in token_arr if token not in stop_words_set]


def default_filter_tokenization(input_str):
    return stem_tokens(filter_stop_tokens(tokenize_str(input_str.lower())))


def tokenize_document(para):
    return default_filter_tokenization(para)


def tokenize_question(question):
    # drop wh-words
    qtokens = default_filter_tokenization(question)
    return [q for q in qtokens if q.lower() not in WH_WORDS]


def asymm_overlap_score(sub_tokens, super_tokens):
    """
    Compute asymmetric overlap score: sub.intersect(super) / sub
    :param sub_tokens:
    :param super_tokens:
    :return:
    """
    tok1_set = set(sub_tokens)
    tok2_set = set(super_tokens)
    if len(tok1_set) and len(tok2_set):
        return len(tok1_set.intersection(tok2_set)) / len(tok1_set)
    return 0.0


def overlap_score(tokens1, tokens2):
    tok1_set = set(tokens1)
    tok2_set = set(tokens2)
    if len(tok1_set) and len(tok2_set):
        return len(tok1_set.intersection(tok2_set)) / len(tok1_set.union(tok2_set))
    return 0.0


def overlaps(str1, str2):
    return overlap_score(default_filter_tokenization(str1), default_filter_tokenization(str2)) > 0.0


def pos_words(text: str, accept_pos: List[str]):
    global spacy_nlp
    global parse_cache
    if spacy_nlp is None:
        import spacy
        print("Initializing spacy")
        spacy_nlp = spacy.load('en_core_web_sm')
    if text in parse_cache:
        doc = parse_cache[text]
    else:
        doc = spacy_nlp(text)
        parse_cache[text] = doc
    pos_words = []
    for token in doc:
        if token.pos_ in accept_pos:
            pos_words.append(token.text)
    return pos_words


def reset_parse_cache():
    global parse_cache
    parse_cache.clear()


def np_chunks(text: str):
    global spacy_nlp
    if spacy_nlp is None:
        import spacy
        spacy_nlp = spacy.load('en_core_web_sm')
    if text in parse_cache:
        doc = parse_cache[text]
    else:
        doc = spacy_nlp(text)
        parse_cache[text] = doc
    return list(doc.noun_chunks)


def is_stopword(word: str):
    return word.lower() in stop_words_set or word in string.punctuation
