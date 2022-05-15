import json
import math
from sklearn_crfsuite import metrics
from tabulate import tabulate

from .entity import *

def evaluate_labels(y_true, y_pred):
    """
    evaluates the sequence model label prediction
    :param y_true: list of list of true labels, each list of labels represent one document
    :param y_pred: list of list of predicted labels, each list of labels represent one document
    :return: nothing
    """
    print(metrics.flat_classification_report(
        y_true, y_pred, labels=['B-SEN', 'E-SEN', 'S-SEN', 'B-IT', 'E-IT', 'O'], digits=3
    ))

def mean(scorelist):
    """
    computes the mean of a given list of values
    as baseline we used this script: https://github.com/finsbd/finsbd2/blob/master/evaluate.py of the finsbd task
    :param scorelist: list of floats
    :return: mean of the givenlist
    """
    # ignore nan value
    scorelist = [e for e in scorelist if not math.isnan(e)]
    if len(scorelist) == 0:
        # all values were nan or list was empty
        return float("nan")
    else:
        # mean
        return sum(scorelist) / len(scorelist)

def evaluate_entities_doc(y_true, y_pred):
    """
    as baseline we used this script: https://github.com/finsbd/finsbd2/blob/master/evaluate.py of the finsbd task
    a entity is predicted right if the start and the end index in the relative document text is right
    :param y_true: the true reference classes of a document as a list of triplets (classId, startId, endId)
    :param y_pred: the predicted classes of a document as a list of triplets (classId, startId, endId)
    :return: a dictionary with an score entry (precision, recall, f1-score) for each entitytype/classid
    """
    scores = {}
    for enttype in ["e_1", "e_2", "item", "e_3", "e_4", "e_5"]:
        true_pairs = []
        pred_pairs = []
        for ent in y_true:
            if ent['classId'] == enttype or enttype == "item" and ent['classId'] in ["e_3", "e_4", "e_5"]:
                true_pairs.append((ent['start'], ent['end']))
        for ent in y_pred:
            if ent['classId'] == enttype or enttype == "item" and ent['classId'] in ["e_3", "e_4", "e_5"]:
                pred_pairs.append((ent['start'], ent['end']))
        if true_pairs or pred_pairs:

            false_negative_list = true_pairs.copy()
            false_positive_list = pred_pairs.copy()

            true_positives = 0
            for pair in pred_pairs:
                if pair in true_pairs:
                    true_positives += 1
                    false_negative_list.remove(pair)
                    false_positive_list.remove(pair)

            # false negative and false positive
            false_negative = len(false_negative_list)
            false_positive = len(false_positive_list)

            # sanity check
            assert len(true_pairs) == len(false_negative_list) + true_positives
            assert len(pred_pairs) == len(false_positive_list) + true_positives

            # precision
            precision = true_positives / (true_positives + false_positive) if (true_positives + false_positive) else 0

            # recall
            recall = true_positives / (true_positives + false_negative) if (true_positives + false_negative) else 0

            # f1 score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        else:
            # return nan which means label does not exist in the document
            precision = float("nan")
            recall = float("nan")
            f1_score = float("nan")
        entdict = {"e_1": 'sentence', "e_2": 'list', "e_3": 'item1', "e_4": 'item2', "e_5": 'item3', "item": "item"}

        scores[entdict[enttype]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
        }
    return scores

def evaluate_entites(test_pathlist, y_pred):
    """
    print for each document the precision, recall and f1-score and the mean score of all documents

    :param y_true: list of lists of triplets (classId, startId, endId) - the true reference classes
    :param y_pred: list of lists of triplets (classId, startId, endId) - the predicted classes
    """
    scores = []
    y_pred = create_pred_eval_data(y_pred)
    y_true = create_test_eval_data(test_pathlist)
    for (true_doc, pred_doc) in zip(y_true, y_pred):
        scores.append(evaluate_entities_doc(true_doc, pred_doc))
    for i in range(len(scores)):
        print("Score document " + str(i + 1) + ":")
        score_doc = scores[i]
        header = ["entity", "precision", "recall", "f1-score"]
        table = []
        for enttype in ["sentence", "list", "item", "item1", "item2", "item3"]:
            score_dict = score_doc[enttype]
            table.append([enttype, score_dict["precision"], score_dict["recall"], score_dict["f1"]])
        prec_mean = mean([score_doc[k]["precision"] for k in ["sentence", "list", "item"]])
        rec_mean = mean([score_doc[k]["recall"] for k in ["sentence", "list", "item"]])
        f1_mean = mean([score_doc[k]["f1"] for k in ["sentence", "list", "item"]])
        table.append(['macro avg / mean after segmentation', prec_mean, rec_mean, f1_mean])
        prec_mean = mean([score_doc[k]["precision"] for k in ["sentence", "list", "item1", "item2", "item3"]])
        rec_mean = mean([score_doc[k]["recall"] for k in ["sentence", "list", "item1", "item2", "item3"]])
        f1_mean = mean([score_doc[k]["f1"] for k in ["sentence", "list", "item1", "item2", "item3"]])
        table.append(['macro avg / mean after hierarchy det.', prec_mean, rec_mean, f1_mean])
        print(tabulate(table, headers=header) + '\n')
    print("Macro avg / mean score of all " + str(len(scores)) + " documents: ")
    table = []
    for enttype in ["sentence", "list", "item", "item1", "item2", "item3"]:
        ent_table = [enttype]
        for score in ["precision", "recall", "f1"]:
            ent_table.append(mean([scores[i][enttype][score] for i in range(len(scores))]))
        table.append(ent_table)
    mean_table = ['macro avg / mean after segmentation']
    for i in [1,2,3]:
        mean_table.append(mean([table[j][i] for j in [0,1,2]]))
    table.append(mean_table)
    mean_table = ['macro avg / mean after hierarchy det.']
    for i in [1,2,3]:
        mean_table.append(mean([table[j][i] for j in [0,1,3,4,5]]))
    table.append(mean_table)
    print(tabulate(table, headers=header))

def create_test_eval_data(pathlist):
    """
    convert the test data set from the tagtog format
    :param pathlist: list of pathes to the different json files in tagtog format
    :return: list of lists of triplets of the format (classId, startid, endid) where startid and endid
             are the relative indices of the document text
    """
    y = []
    for path in pathlist:
        dictlist = []
        with open(path) as json_file:
            data = json.load(json_file)
        entities = data['entities']
        for ent in entities:
            dictlist.append({'classId': ent['classId'], 'start': ent['offsets'][0]['start'], 'end': ent['offsets'][0]['start'] + len(ent['offsets'][0]['text'])})
        y.append(dictlist)
    return y

def create_pred_eval_data(y_pred):
    """
    transform the the predicted data into a list of list of dictionaries with a dict for each entity in a doc
    :param y_pred: list of list of entites from the segmentation
    :return: list of lists of triplets of the format (classId, startid, endid) where startid and endid
             are the relative indices of the document text. Format for the evaluation of the segmentation
    """
    y = []
    for doc in y_pred:
        dictlist = []
        for entity in doc:
            classId = ''
            if isinstance(entity, Sentence):
                classId = 'e_1'
            elif isinstance(entity, List):
                classId = 'e_2'
            elif isinstance(entity, Item1):
                classId = 'e_3'
            elif isinstance(entity, Item2):
                classId = 'e_4'
            elif isinstance(entity, Item3):
                classId = 'e_5'
            dictlist.append({'classId' : classId, 'start': entity.startid, 'end': entity.endid})
        y.append(dictlist)
    return y
