import json
import math
from tabulate import tabulate
from nltk.tokenize import RegexpTokenizer
import sys
sys.path.append("..")
from list_identification.postprocessing.entity import List, Item1, Item2
from grammarchecker.grammar_checker import GrammarChecker
from list_complementation.complementation import get_sentences

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

def evaluate_complementation(y_true, y_pred):
    """
    computes based on the token list of the complemented sentences the different performance metrics for both classes
    :param y_true: list of tuples, each tuple (completable, noncompletable) represents a document, where completable
        stores completable sentences and noncompletable stores all non-completable sentences as tokenlists
    :param y_pred: list of tuples, each tuple (completable, noncompletable) represents a document, where completable
        stores completable sentences and noncompletable stores all non-completable sentences as tokenlists
    :return: nothing
    """
    dict = []
    header = ["entity", "precision", "recall", "f1-score"]
    for (truetupel, predtupel) in zip(y_true, y_pred):
        #0: completable
        #1: noncompletable
        tmpdict = []
        for i in range(2):
            true_tupel = truetupel[i]
            pred_tupel = predtupel[i]
            if true_tupel or pred_tupel:
                false_negative_list = true_tupel.copy()
                false_positive_list = pred_tupel.copy()
                true_positives = 0
                for tupel in pred_tupel:
                    if tupel in true_tupel:
                        true_positives += 1
                        false_negative_list.remove(tupel)
                        false_positive_list.remove(tupel)
                false_negative = len(false_negative_list)
                false_positive = len(false_positive_list)

                # sanity check
                assert len(true_tupel) == len(false_negative_list) + true_positives
                assert len(pred_tupel) == len(false_positive_list) + true_positives

                # precision
                precision = true_positives / (true_positives + false_positive) if (
                            true_positives + false_positive) else 0

                # recall
                recall = true_positives / (true_positives + false_negative) if (true_positives + false_negative) else 0

                # f1 score
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

                scoredict = {"precision": precision, "recall": recall, "f1": f1_score}
                tmpdict.append(scoredict)
        dict.append(tmpdict)
    for i in range(len(y_true)):
        # configuration for output
        tmptable = []
        print("Score document " + str(i + 1) + ":")
        scoredict = dict[i]
        tmptable.append(["Completable", scoredict[0]["precision"], scoredict[0]["recall"], scoredict[0]["f1"]])
        tmptable.append(["Not Completable", scoredict[1]["precision"], scoredict[1]["recall"], scoredict[1]["f1"]])
        li = [scoredict[j]["precision"] for j in [0,1]]
        prec_mean = mean([scoredict[j]["precision"] for j in [0,1]])
        rec_mean = mean([scoredict[j]["recall"] for j in [0, 1]])
        f1_mean = mean([scoredict[j]["precision"] for j in [0, 1]])
        tmptable.append(['macro avg / mean', prec_mean, rec_mean, f1_mean])
        print(tabulate(tmptable, headers=header) + '\n')
    print("Macro avg / mean score of all " + str(len(y_true)) + " documents: ")
    table = []
    for i in [0,1]:
        category = "Completable"
        if i == 1:
            category = "Not completable"
        ent_table = [category]
        for score in ["precision", "recall", "f1"]:
            ent_table.append(mean([dict[k][i][score] for k in range(len(dict))]))
        table.append(ent_table)
    mean_table = ['macro avg / mean']
    for i in [1,2,3]:
        mean_table.append(mean([table[j][i] for j in [0,1]]))
    table.append(mean_table)
    print(tabulate(table, headers=header))

def create_ref(pathlist):
    """
    uses the list complementation approach to create reference data on the dataset
    :param pathlist to the annotated test data
    :return: list of tuples, each tuple represent one document, tuple = (completable, noncompletable)
        both are list of token list, where each token list represents a sentence
    """
    tokenizer = RegexpTokenizer('[a-zA-Z-\']+|[0-9]+|[^a-zA-Z0-9\s]')
    out = []
    gc = GrammarChecker()
    for path in pathlist:
        completable = []
        noncompletable = []
        with open(path) as json_file:
            data = json.load(json_file)
        entities = data["entities"]
        for ent in entities:
            sen1 = tokenizer.tokenize(ent["sentence1"]["text"])
            sen2 = tokenizer.tokenize(ent["sentence2"]["text"])
            clearedsen1 = sen1
            clearedsen2 = sen2
            # check for specific characters to remove
            if clearedsen1[-1] == ":":
                clearedsen1 = clearedsen1[0:-1]
            if clearedsen2[-1] == ';':
                clearedsen2 = clearedsen2[0:-1]
            elif (clearedsen2[-1] in ['and', 'or'] and clearedsen2[-2] == ';') or (clearedsen2[-1] == ';' and clearedsen2[-2] in ['and', 'or']):
                clearedsen2 = clearedsen2[0:-2]
            # do the main grammar check
            if gc.grammarcheck(clearedsen1 + clearedsen2):
                completable.append((sen1,sen2))
            else:
                noncompletable.append((sen1,sen2))
        out.append((completable, noncompletable))
    return out

def create_true_data(pathlist):
    """
    creates the true data for the evaluation

    :param pathlist: path to the testset data
    :return: list of tuples, where each tuple represents the reference data from the respective document
        each tuple consists of (completable,noncompletable), both are lists with each entry a token list
    """
    tokenizer = RegexpTokenizer('[a-zA-Z-\']+|[0-9]+|[^a-zA-Z0-9\s]')
    out = []
    for path in pathlist:
        completable = []
        noncompletable = []
        with open(path) as json_file:
            data = json.load(json_file)
        entities = data["entities"]
        for ent in entities:
            sen1 = tokenizer.tokenize(ent["sentence1"]["text"])
            sen2 = tokenizer.tokenize(ent["sentence2"]["text"])
            if ent["completable"]:
                completable.append((sen1,sen2))
            else:
                noncompletable.append((sen1, sen2))
        out.append((completable,noncompletable))
    return out

def complementation_helper(introtokens, entity):
    """
    rebuild the recursive complementationhelper from complementation.py for evaluation purposes
    return all completable and non completable sentences for a (sub-)list
    :param introtokens: list of tokens
    :param entity: Item object
    :return: tupel of two list (completable, noncompletable), completable represents all completable sentences
        noncompletable represents all non-completable sentences
    """
    completable=[]
    noncompletable = []
    gc = GrammarChecker()
    children = entity.children
    for child in children:
        sentences = get_sentences(child)
        # check if there are children
        if (isinstance(child, Item1) or isinstance(child, Item2)) and len(child.children) != 0:
            if len(sentences) == 1:
                tmpcompl, tmpnoncompl = complementation_helper(introtokens + sentences[0], child)
                completable.extend(tmpcompl)
                noncompletable.extend(tmpnoncompl)
            else:
                if gc.grammarcheck(introtokens) and gc.grammarcheck(sentences[0]):
                    noncompletable.append((introtokens, child.bulletlist + sentences[0]))
                elif gc.grammarcheck(introtokens + sentences[0]):
                    completable.append((introtokens, child.bulletlist + sentences[0]))
                else:
                    noncompletable.append((introtokens, child.bulletlist + sentences[0]))
                tmpcompl, tmpnoncompl = complementation_helper(sentences[-1],child)
                completable.extend(tmpcompl)
                noncompletable.extend(tmpnoncompl)
        else:
            if gc.grammarcheck(introtokens) and gc.grammarcheck(sentences[0]):
                noncompletable.append((introtokens, child.bulletlist + sentences[0]))
            elif gc.grammarcheck(introtokens + sentences[0]):
                completable.append((introtokens, child.bulletlist + sentences[0]))
            else:
                noncompletable.append((introtokens, child.bulletlist + sentences[0]))
    return completable, noncompletable

def create_pred_data(y_pred):
    """
    to evaluate the list complementation approach with predicted data we had to reimplement the complementation
    methods and adapt them for the evaluation to create the completable list and the non complementable list
    :param y_pred: list of list of entities, each list represent a document
    :return: list of tupeles of two list (completable, noncompletable), completable represents all completable sentences
        noncompletable represents all non-completable sentences, each tuple represent a document
    """
    out = []
    for doc in y_pred:
        completable = []
        noncompletable = []
        for ent in doc:
            if isinstance(ent, List):
                tmpcompl, tmpnoncompl = complementation_helper(ent.introtokens, ent)
                completable.extend(tmpcompl)
                noncompletable.extend(tmpnoncompl)
        out.append((completable,noncompletable))
    return out
