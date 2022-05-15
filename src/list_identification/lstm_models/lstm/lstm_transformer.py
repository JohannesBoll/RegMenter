from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from hashlib import md5

class LSTMTransformer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    # creates the feature dict of a given token
    def __feature(self, sequence, index, word_index,):
        special_characters = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        return {
            'word_index': word_index,
            'is_ascii': sequence[index].isascii(),
            'possible_item_start': True if ((index < len(sequence) - 1 and
                (sequence[index].isdigit() or sequence[index].isalpha()) and sequence[index + 1] == ')')
                or sequence[index] == '-' or sequence[index] == '\uf0a7') else False,
            'length': len(sequence[index]),
            'is_special': True if (any(c in special_characters for c in sequence[index]) or not (
                sequence[index].isascii())) else False,
            'is_numeric': sequence[index].isdigit(),
            'is_capitalized': sequence[index][0].isupper(),
            'possible_sentence_start': True if (index - 1 > 0 and sequence[index-1] in ['.', '!', '?'] and sequence[index][0].isupper()) else False
        }

    # iterator over list of tokens of document
    def __iterator(self, sequence, wordidlist):
        # sets the window size
        dicts = []
        for index in range(len(sequence)):
            dicts.append(self.__feature(sequence, index, wordidlist[index]))
        return dicts

    # expects a nested list with an entry for each document to transform
    # the documents are a list of tokens
    def transform(self, documents):
        print("Starting feature transformation...")
        out = []
        i = 1
        v = DictVectorizer(sparse=False)
        nested_tokens = [item for sublist in documents for item in sublist]
        words = set(nested_tokens)
        vocab_size = len(words)
        # integer encode the document
        def hash_function(w):
            return int(md5(w.encode()).hexdigest(), 16)
        n = round(vocab_size * 1.3)
        result = [(hash_function(w) % (n - 1) + 1) for w in nested_tokens]
        startid = 0
        for document in documents:
            wordidlist = result[startid:(startid + len(document))]
            out.append(v.fit_transform(self.__iterator(document, wordidlist)))
            print("Transformed " + str(i) + " of " + str(len(documents)) + " documents")
            i += 1
            startid += len(document)
        return out
