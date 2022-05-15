import joblib

from list_identification.parser.textextraction import extract_text
from list_identification.tokenizer.tokenizer import Tokenizer
from list_identification.postprocessing.SBDPostprocessing import label_completion, segment_tokens, compute_hierarchy
from list_complementation.complementation import complementation

def predict(pathlist):
    """
    segments a given list of paths to documents into a list of token lists representing
    :param pathlist: list of pathes to PDF documents
    :return: list of list of token list, token list represent a sentence, the list of token lists represents all segmented sentences
    """
    print("Starting sentence boundary detection")
    textlist = extract_text(pathlist)
    tokenzier = Tokenizer('[a-zA-Z-\']+|[0-9]+|[^a-zA-Z0-9\s]')
    tokenlist = []
    for doc in textlist:
        tokenlist.append((doc,tokenzier.tokenize(doc)))
    sequence_model = joblib.load(# TODO)
    y_pred = sequence_model.predict(tokenlist)
    y_pred = label_completion(y_pred)

    # do segmentation and hierarchy detection
    y_pred = segment_tokens(tokenlist, y_pred)
    y_pred = compute_hierarchy(y_pred)
    sentences = complementation(y_pred)
    print("Finished sentence boundary detection")
    # return all segmented sentences of all documents
    return sentences


if __name__ == "__main__":
    # define a pathlist of PDF documents to segment
    pathlist_doc = [
        # TODO
    ]
    predict(pathlist_doc)