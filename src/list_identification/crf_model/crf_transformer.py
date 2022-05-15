from sklearn.base import BaseEstimator, TransformerMixin

class CRFTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer based on the sklearn Transformer to transform the features

    :param windowsize: indicates the windowssize of features around each token
    """

    def __init__(self, windowsize=5):
        self.windowsize = windowsize

    def fit(self, x, y=None):
        return self

    # creates the feature dict of a given token
    def __feature(self, tokenseq, tokenId, text, offset):
        """
        transform the token to a feature dict for the token

        :param tokenseq: List of the tokens of the document
        :param tokenId: Index of the actual token in tokenseq
        :param text: String of the text of the document
        :param offset: offset inside the given window

        :return:
        Feature dict for a given Token
        """
        special_characters = "#$%&'*+/<=>@[\]^_`{|}~"
        token = tokenseq[tokenId][0]
        if token not in special_characters:
            special = 'No'
        elif token in [".", "!", "?"]:
            special = 'End'
        elif token == "(":
            special = "Open"
        elif token == ")":
            special = "Close"
        else:
            special = "S"
        startid = tokenseq[tokenId][1][0]
        endid = tokenseq[tokenId][1][1]
        newlines_before, newlines_after, spaces_before, spaces_after, sn_before, sn_after = 0, 0, 0, 0, 0, 0
        i1, i2, i3, i4, i5, i6 = startid - 1, endid, startid - 1, endid, startid - 1, endid
        while i1 > 0 and text[i1] == '\n':
            newlines_before += 1
            i1 -= 1
        while i2 < len(text) and text[i2] == '\n':
            newlines_after += 1
            i2 += 1
        while i3 > 0 and text[i3] == ' ':
            spaces_before += 1
            i3 -= 1
        while i4 < len(text) and text[i4] == ' ':
            spaces_after += 1
            i4 += 1
        while i5 > 0 and text[i5] in [' ', '\n']:
            sn_before += 1
            i5 -= 1
        while i6 < len(text) and text[i6] in [' ', '\n']:
            sn_after += 1
            i6 += 1
        return {
            # basic features about the main token
            'bias': 1.0,
            'token[' + str(offset) + ']': token,
            'lowercase[' + str(offset) + ']': token.lower(),
            'length[' + str(offset) + ']': len(token),
            'is_ascii[' + str(offset) + ']': token.isascii(),
            'is_numeric[' + str(offset) + ']': token.isdigit(),
            'is_capitalized[' + str(offset) + ']': token[0].isupper(),
            'sentence_terminating_char[' + str(offset) + ']': token in ['!', '?', '.'],
            'colon[' + str(offset) + ']': token == ':',
            'comma[' + str(offset) + ']': token == ',',
            'semicolon[' + str(offset) + ']': token == ';',
            'punctuation[' + str(offset) + ']': token == ':',
            'quotation_marks[' + str(offset) + ']': token == '\"',
            'brackets[' + str(offset) + ']': token in ['(', ')'],
            'is_special[' + str(offset) + ']': True if (any(c in special_characters for c in token)) else False,
            '–[' + str(offset) + ']': token == '–',
            '-[' + str(offset) + ']': token == '-',
            '•[' + str(offset) + ']': token == '•',
            '\ufffd[' + str(offset) + ']': token == '�',
            '\uf0a7[' + str(offset) + ']': token == '\uf0a7',

            # features from the sorrounding context
            'space_before[' + str(offset) + ']': True if spaces_before > 0 else False,
            'spaces_before[' + str(offset) + ']': spaces_before,
            'space_after[' + str(offset) + ']': True if spaces_after > 0 else False,
            'spaces_after[' + str(offset) + ']': spaces_after,
            'newline_before[' + str(offset) + ']': True if newlines_before > 0 else False,
            'newlines_before[' + str(offset) + ']': newlines_before,
            'newline_after[' + str(offset) + ']': True if newlines_after > 0 else False,
            'newlines_after[' + str(offset) + ']': newlines_after,
            'sn_before[' + str(offset) + ']': sn_before,
            'sn_after[' + str(offset) + ']': sn_after,

            'token.[' + str(offset) + ']': True if tokenId + 1 < len(tokenseq) and tokenseq[tokenId + 1][0] == '.' and len(tokenseq[tokenId][0]) <= 4 else False,
            'token)[' + str(offset) + ']': True if tokenId + 1 < len(tokenseq) and tokenseq[tokenId + 1][0] == ')' and len(
                tokenseq[tokenId][0]) <= 4 else False,
            '(token)[' + str(offset) + ']': True if tokenId + 1 < len(tokenseq) and tokenseq[tokenId + 1][0] == ')' and tokenId - 1 > 0 and tokenseq[tokenId - 1][0] == '('and  len(
                tokenseq[tokenId][0]) <= 4 else False,
            'special[' + str(offset) + ']': special
        }

    # iterator over list of tokens of document
    def __iterator(self, document):
        """
        iterate over all tokens and compute for the actual token and all tokens in the given window around the token features

        :param document: List of tuples of (text, tokens_with_span) representing the given document

        :return:
        dicts: List of feature-dictionaries, one dictionary for each token
        """
        # sets the window
        window = list(range(-self.windowsize, self.windowsize + 1))
        doc_text = document[0]
        sequence = document[1]
        dicts = []
        for index in range(len(sequence)):
            tmp_dict = {}
            # add for the token and all tokens in the window the features
            for offset in window:
                if index + offset < 0:
                    tmp_dict.update({str(offset): 'BOS'})
                elif index + offset >= len(sequence) - 1:
                    tmp_dict.update({str(offset): 'EOS'})
                else:
                    tmp_dict.update(self.__feature(sequence, index + offset, doc_text,offset))
            dicts.append(tmp_dict)
        return dicts

    def transform(self, documents):
        """
        iterate over all documents and tranform them to feature dicts

        :param documents: List of lists of tuples of (text, tokens_with_span); each list representing a document
            text: string of the document text
            tokens_with_span: a list of tuples of the structure (token, (startid, endid) where the ids are relative to text

        :return:
        out: list of lists of dicts
            Feature dicts for several documents (in a python-crfsuite format).
        """
        #print("Starting feature transformation with windowsize " + str(self.windowsize))
        out = []
        i = 1
        for document in documents:
            out.append(self.__iterator(document))
            #print("Transformed " + str(i) + " of " + str(len(documents)) + " documents")
            i += 1
        return out
