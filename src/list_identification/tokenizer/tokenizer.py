from nltk.tokenize import RegexpTokenizer

class Tokenizer:
    def __init__(self, regex):
        self.tokenizer = RegexpTokenizer(regex)

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        spans = self.tokenizer.span_tokenize(text)
        return list(zip(tokens, spans))