import spacy
from spacy.tokens import Doc

class GrammarChecker:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md",  disable=["lemmatizer", "ner"])


    def grammarcheck(self, tokens):
        """
        Check if a given tokensequence representing a sentence contains a subject and a finite verb
        No check for capital letter and punctuation, since this is already done in the list completion
        :param tokens: list of tokens representing a possible sentence
        :return: Boolean if a sentence is complete or not
        """

        doc = Doc(self.nlp.vocab, tokens)
        hasFinVerb = False
        hasSubject = False
        for token in self.nlp(doc):
            if 'VerbForm=Fin' in token.morph:
                hasFinVerb = True
            if token.dep_ in ["csubj", "nsubj"]:
                hasSubject = True
        if (hasFinVerb and hasSubject) or len(tokens) == 0:
            return True
        return False

