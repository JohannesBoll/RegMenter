import sys
sys.path.append("..")
from list_identification.postprocessing.entity import *
from grammarchecker.grammar_checker import GrammarChecker
import joblib


def notifcation(parent, child, introtokens):
    """
    notify the user about non complementable items
    """
    print("The following entities could not be completed:")
    print("Type: " + type(parent).__name__ + '/' + type(child).__name__)
    print("Introduction Sentence:")
    print("Relative start index: " + str(parent.startid))
    print("Relative end index: " + str(parent.endid))
    print(introtokens)
    print("Listitem:")
    print("Relative start index: " + str(child.startid))
    print("Relative end index: " + str(child.endid))
    print(child.tokenlist)
    print("\n")


def get_sentences(child):
    """
    segment the tokens of a given child into sentences
    :param child: Item object
    :return: list of token list each list representing a segmented sentence
    """
    tokenlist = child.tokenlist[len(child.bulletlist):]
    if len(tokenlist) == 1:
        return [tokenlist]
    elif len(tokenlist) == 0:
        return [child.bulletlist]
    text = ' '.join(tokenlist)
    tokenlist_with_spans = []
    start = 0
    for token in tokenlist:
        end = start + len(token)
        tokenlist_with_spans.append((token, (start, end)))
        start = end + 1
    pipeline = joblib.load(#TODO)
    y_pred = pipeline.predict([(text, tokenlist_with_spans)])[0]
    # change item labels to sentence labels
    for i in range(len(y_pred)):
        if y_pred[i] == 'B-IT':
            y_pred[i] = 'B-SEN'
        elif y_pred[i] == 'E-IT':
            y_pred[i] = 'E-SEN'
    sentences = []
    start_tag = y_pred[0]
    start_tag_id = 0
    # check for missing end sentence tokens
    for i in range(len(y_pred)):
        if (i + 1 < len(y_pred) and y_pred[i + 1] in ['B-SEN', 'S-SEN']) or i == len(y_pred) - 1:
            if y_pred[i] == 'O':
                if start_tag == 'B-SEN':
                    y_pred[i] = 'E-SEN'
                elif start_tag == 'O':
                    y_pred[start_tag_id] = 'B-SEN'
                    y_pred[i] = 'E-SEN'
            if i + 1 < len(y_pred):
                start_tag = y_pred[i + 1]
                start_tag_id = i + 1
    start_tag = y_pred[0]
    start_tag_id = 0
    # check for missing end sentence tokens
    for i in range(len(y_pred)):
        if y_pred[i] in ['E-SEN', 'S-SEN']:
            if start_tag == 'O':
                if y_pred[i] == 'E-SEN':
                    y_pred[start_tag_id] = 'B-SEN'
            if i + 1 < len(y_pred):
                start_tag = y_pred[i + 1]
                start_tag_id = i + 1
    # check for special cases at start of the sequence
    if len(y_pred) > 1 and y_pred[1] == 'B-SEN':
        y_pred[0] = 'B-SEN'
        y_pred[1] = 'O'
        if len(y_pred) == 2:
            y_pred[0] = 'B-SEN'
            y_pred[1] = 'E-SEN'
    # finally segment
    for i in range(len(y_pred)):
        if y_pred[i] == 'S-SEN':
            sentences.append([tokenlist[i]])
        elif y_pred[i] == 'B-SEN':
            start_id = i
        elif y_pred[i] == 'E-SEN':
            sentences.append(tokenlist[start_id:i+1])
    return sentences

def complementation_helper(introtokens, entity):
    """
    recursive procedure to complement all different hierachies of a List
    :param introtokens: introductory sentence of a (sub-)List
    :param entity: child of the next higher hierarchy (Item object)
    :return: List of token lists, where each list represents a segmented sentence
    """
    out = []
    gc = GrammarChecker()
    children = entity.children
    if len(introtokens) > 0 and introtokens[-1] == ':':
        introtokens = introtokens[0:-1]
    for child in children:
        sentences = get_sentences(child)
        # remove at -1, because they occour at the end
        if sentences[-1][-1] == ';':
            sentences[0] = sentences[-1][0:-1]
        elif (sentences[-1][-1] in ['and', 'or'] and sentences[0][-2] == ';') or (sentences[0][-2] in ['and', 'or'] and sentences[-1][-1] == ';'):
            sentences[0] = sentences[-1][0:-2]
        # check if there are children
        if (isinstance(child, Item1) or isinstance(child, Item2)) and len(child.children) != 0:
            if len(sentences) == 1:
                out.append(complementation_helper(introtokens + sentences[0], child))
            else:
                if gc.grammarcheck(introtokens) and gc.grammarcheck(sentences[0]):
                    out.append(introtokens)
                    out.append(sentences[0])
                elif gc.grammarcheck(introtokens + sentences[0]):
                    out.append(introtokens + sentences[0])
                else:
                    notifcation(entity, child, introtokens)
                    out.append(introtokens)
                    out.append(sentences[0])
                numSentences = len(sentences) - 2
                if numSentences != 0:
                    out.append(sentences[1:1+numSentences])
                out.append(complementation_helper(sentences[-1],child))
        else:
            if gc.grammarcheck(introtokens) and gc.grammarcheck(sentences[0]):
                out.append(introtokens)
                out.append(sentences[0])
            elif gc.grammarcheck(introtokens + sentences[0]):
                out.append(introtokens + sentences[0])
            else:
                notifcation(entity, child, introtokens)
                out.append(introtokens)
                out.append(sentences[0])
            if len(sentences) > 1:
                out.append(sentences[1:-1])
    return out

def complementation(documents):
    """
    Transform an List of entites consisting of sentences, lists and items to a list of sentences for further processing
    inside a nlp pipeline
    :param documents: List of List of entites consisting of sentences, lists and items, each list represent a doc
    :return: List of Lists of tokens representing each a sentence, which can be used in the further nlp pipeline
    """
    # first off all iterate over all entities and find the lists
    sentences = []
    for entities in documents:
        doc_sentences = []
        for ent in entities:
            if isinstance(ent,List):
                doc_sentences.append(complementation_helper(ent.introtokens, ent))
            elif isinstance(ent, Sentence):
                doc_sentences.append(ent.tokenlist)
        sentences.append(doc_sentences)
    return sentences
