from .entity import *

def completion0(y):
    """
    complete sequences without closing E- Tag

    :param y: List of predicted labels of a document
    :return: List of completed labels of the document
    """
    start_tag = y[0]
    start_tag_id = 0
    for i in range(len(y)):
        if i + 1 < len(y) and y[i + 1] in ['B-SEN', 'B-IT', 'S-SEN']:
            if y[i] == 'O':
                if start_tag == 'B-IT':
                    y[i] = 'E-IT'
                elif start_tag == 'B-SEN':
                    y[i] = 'E-SEN'
                elif start_tag == 'O':
                    y[start_tag_id] = 'B-SEN'
                    y[i] = 'E-SEN'
            start_tag = y[i + 1]
            start_tag_id = i + 1
    return y

def completion1(y):
    """
    complete sequence with missing starting B- Tag

    :param y: List of predicted labels of a document
    :return: List of completed labels of the document
    """
    start_tag = y[0]
    start_tag_id = 0
    for i in range(len(y)):
        if y[i] in ['E-SEN', 'E-IT', 'S-SEN']:
            if start_tag == 'O':
                if y[i] == 'E-IT':
                    y[start_tag_id] = 'B-IT'
                elif y[i] == 'E-SEN':
                    y[start_tag_id] = 'B-SEN'
            if i + 1 < len(y):
                start_tag = y[i + 1]
                start_tag_id = i + 1
    return y

def completion2(y):
    """
    B-IT are trustworthy predictions, a sequence with a starting B-IT and a ending E-SEN is transformed to an item sequence

    :param y: List of predicted labels of a document
    :return: List of completed labels of the document
    """
    start_tag = y[0]
    start_tag_id = 0
    for i in range(len(y)):
        if y[i] in ['E-SEN', 'E-IT', 'S-SEN']:
            if y[i] == 'E-SEN' and start_tag == 'B-IT':
                e_sen = i
                h = i
                while h < len(y) and h < start_tag_id + 300 and y[h] not in ['B-IT', 'E-IT']:
                    h += 1
                if y[h] == 'E-IT':
                    for h in range(start_tag_id + 1, h):
                        y[h] = 'O'
                else:
                    if get_next_startlabel(start_tag_id, y) == 'B-IT' or get_last_startlabel(start_tag_id,y) == 'B-IT':
                        y[e_sen] = 'E-IT'
                    else:
                        y[start_tag_id] = 'B-SEN'
            if i + 1 < len(y):
                if y[i + 1] in ['B-SEN', 'B-IT', 'S-SEN', 'O']:
                    start_tag = y[i + 1]
                    start_tag_id = i + 1
    return y

def completion3(y):
    """
    handle sequence with beginning of B-SEN and ending of E-IT

    :param y: List of predicted labels of a document
    :return: List of completed labels of the document
    """
    start_tag = y[0]
    start_tag_id = 0
    for i in range(len(y)):
        if y[i] in ['E-SEN', 'E-IT', 'S-SEN']:
            #decide on the sorrounding begin and end labels
            if y[i] == 'E-IT' and start_tag == 'B-SEN':
                if get_last_startlabel(start_tag_id, y) == 'B-IT' and  get_next_startlabel(i, y) == 'B-IT':
                    y[start_tag_id] = 'B-IT'
                else:
                    y[i] = 'E-SEN'
            if i + 1 < len(y):
                if y[i + 1] in ['B-SEN', 'B-IT', 'S-SEN', 'O']:
                    start_tag = y[i + 1]
                    start_tag_id = i + 1
    return y

def completion4(y):
    """
    remove false B- or S- labels inside a sequence

    :param y: List of predicted labels of a document
    :return: List of completed labels of the document
    """
    start_tag = y[0]
    start_tag_id = 0
    for i in range(len(y)):
        if 'B-' in start_tag and ('B-' in y[i] or 'S-' in y[i]) and i != start_tag_id:
            y[i] = 'O'
        if y[i] in ['E-SEN', 'E-IT', 'S-SEN']:
            if i + 1 < len(y):
                if y[i + 1] in ['B-SEN', 'B-IT', 'S-SEN', 'O']:
                    start_tag = y[i + 1]
                    start_tag_id = i + 1
    return y

def label_completion(y):
    """
    for each document run different completion steps on the labels

    :param y: List of list of predicted labels; each list represents a document
    :return: List of completed list of predicted labels; each list represents a document
    """
    for i in range(len(y)):
        y[i] = completion0(y[i])
        y[i] = completion1(y[i])
        y[i] = completion2(y[i])
        y[i] = completion3(y[i])
        y[i] = completion4(y[i])
    return y

def get_last_startlabel(index, labels):
    """
    determine the last B- label before the label at the position index

    :param index: Index from which is searched in descending order
    :param labels: List of labels
    :return: the last B- label before the label at the position index
    """
    while (index - 1 > 0):
        index -= 1
        label = labels[index]
        if 'B-' in label or 'S-' in label:
            return label
    return None

def get_last_endlabel(index, labels):
    """
    determine the last E- label before the label at the position index

    :param index: Index from which is searched in descending order
    :param labels: List of labels
    :return: the last E- label before the label at the position index
    """
    while (index - 1 > 0):
        index -= 1
        label = labels[index]
        if 'E-' in label or 'S-' in label:
            return label
    return None

def get_next_startlabel(index, labels):
    """
    determine the next B- label after the label at the position index

    :param index: Index from which is searched
    :param labels: List of labels
    :return: the next B- label after the label at the position index
    """
    while(index + 1 < len(labels)):
        index += 1
        label = labels[index]
        if 'B-' in label or 'S-' in label:
            return label
    return None

def get_next_endlabel(index, labels):
    """
    determine the next E- label after the label at the position index

    :param index: Index from which is searched
    :param labels: List of labels
    :return: the next E- label after the label at the position index
    """
    while(index + 1 < len(labels)):
        index += 1
        label = labels[index]
        if 'E-' in label or 'S-' in label:
            return label
    return None

def get_bulletpoint(startid, endid, tokenlist):
    """
    compute the bulletpoint in form of a list of tokens and returns it, if no bulletpoint is detected return empty list

    :param startid: Start index of the given sequence in the tokenlist
    :param endid: End index of the given sequence in the tokenlist
    :param tokenlist: List of tokens in the format [(token, tokenspan),...]
    :return:
    """

    # for all combinations with . inside the bulletpoints we assume a individual length of 1 of the alphanumeric chars

    # bulletpoint with an open bracket and closing bracket and a point -> (char.char)
    if startid + 4 <= endid and tokenlist[startid][0] == '(' and len(tokenlist[startid + 1][0]) == 1 and (tokenlist[startid + 1][0].isalpha() or tokenlist[startid + 1][0].isnumeric()) and tokenlist[startid + 2][0] == '.' and len(tokenlist[startid + 3][0]) == 1 and (tokenlist[startid + 3][0].isalpha() or tokenlist[startid + 3][0].isnumeric()) and tokenlist[startid + 4][0] == ')' :
        return [tokenlist[startid][0], tokenlist[startid + 1][0], tokenlist[startid + 2][0]]

    if (startid + 3 <= endid):
        #bulletpoint in the format: char.char)
        if len(tokenlist[startid][0]) == 1 and tokenlist[startid + 1][0] == '.' and len(tokenlist[startid + 2][0]) == 1 and (tokenlist[startid + 2][0].isalpha() or tokenlist[startid + 2][0].isnumeric()) and tokenlist[startid + 3][0] == ')':
            return [tokenlist[startid][0], tokenlist[startid + 1][0], tokenlist[startid + 2][0]]

    if (startid + 2 <= endid):
        # bulletpoint with a point and a closing bracket, e.g. 1.)
        if len(tokenlist[startid][0]) <= 4 and (tokenlist[startid][0].isalpha() or tokenlist[startid][0].isnumeric()) and tokenlist[startid + 1][0] == '.' and tokenlist[startid + 2][0] == ')':
            return [tokenlist[startid][0], tokenlist[startid + 1][0], tokenlist[startid + 2][0]]
        # bulletpoint with an open bracket and closing bracket \([a-zA-Z0-9]{1,3}\), e.g. (aaa)
        if tokenlist[startid][0] == '(' and len(tokenlist[startid + 1][0]) <= 4 and (tokenlist[startid + 1][0].isalpha() or tokenlist[startid + 1][0].isnumeric()) and tokenlist[startid + 2][0] == ')' :
            return [tokenlist[startid][0], tokenlist[startid + 1][0], tokenlist[startid + 2][0]]
        #bulletpoints with in the format: char.char
        if (tokenlist[startid][0].isalpha() or tokenlist[startid][0].isnumeric()) and len(tokenlist[startid][0]) == 1 and tokenlist[startid + 1][0] == '.' and len(tokenlist[startid + 2][0]) == 1 and (tokenlist[startid + 2][0].isalpha() or tokenlist[startid + 2][0].isnumeric()):
            return [tokenlist[startid][0], tokenlist[startid + 1][0], tokenlist[startid + 2][0]]

    # bulletpoint with a closing bracket or point [a-zA-Z0-9]{1,3}[\)|.)]
    if startid + 1 <= endid and len(tokenlist[startid][0]) <= 4 and (tokenlist[startid][0].isalpha() or tokenlist[startid][0].isnumeric()) and (tokenlist[startid + 1][0] == ')' or tokenlist[startid + 1][0] == '.'):
        return [tokenlist[startid][0], tokenlist[startid + 1][0]]

    # defines a bulletpoint without a number or letter in it, can be modified depending on the document domain
    if not tokenlist[startid][0].isascii() or tokenlist[startid][0] in ['~', '-', '+', '#', '$', '*']:
        return [tokenlist[startid][0]]

    #bulletpoint consisting of one token a or 1 (len <= 3 for cases like iii)
    if (tokenlist[startid][0].isnumeric() or tokenlist[startid][0].isalpha()) and len(tokenlist[startid][0]) <= 4:
        return [tokenlist[startid][0]]

    # if no case matches
    return []

def get_sequencelist(tokens, startid, endid):
    """
    get a sublist of a tokenlist with the boundaries startid and endid

    :param tokens: List of tokens in the format [(token, tokenspan),...]
    :param startid: Start index of the sublist to extract from the tokenlist
    :param endid: End index of the sublist to extract from the tokenlist
    :return:
    """
    sequencelist = []
    while (startid <= endid and startid < len(tokens)):
        sequencelist.append(tokens[startid][0])
        startid +=1
    return sequencelist

def segment_tokens(tokens, labels):
    """
    segment/convert the flat predicted label lists of sentences and items to sentences, lists, items
    no hierarchies are considered yet - here only items are assigned to lists

    some variable explanations:
    tokenlist: list of tuples (token, (startid, endid)), where the ids are relative to the string of the document text
    token: the actual token at index i
    entity_start and entity_end: the start and end index relative to the string of the document text
    entity_tokenlist: the list of tokens that belong to a sentence, list or item
    X_token_startid, X_token_endid: start and end index relative to the tokenlist
    intro_tokens: introductions sentence of a list if existent
    itemlist: list of items of a list
    bulletpoint: list of tokens representing a bulletpoint

    :param tokens: List of lists of tuples of (text, tokens_with_span); each list representing a document
            text: string of the document text
            tokens_with_span: a list of tuples of the structure (token, (startid, endid)) where the ids are relative to text

    :param labels: List of lists of labels (strings); each list representing a document
    :return: List of lists of entities (Sentence, List and Item), each list represents an document
    """
    entities = []
    for (doc_tokens, doc_labels) in zip(tokens, labels):
        doc_entities = []
        i = 0
        tokenlist = doc_tokens[1]
        while i < len(doc_labels):
            token = doc_tokens[1][i]
            label = doc_labels[i]
            # check if sentence is a sentence or with a B-SEN or S-SEN and a following B-IT a possible start of list
            # if the second case occurs skip this sentence, because it is later taken into account in the else branch as list start
            if label in ['B-SEN', 'S-SEN']:
                if get_next_startlabel(i, doc_labels) not in ['B-IT', 'S-IT']:
                    if label == 'S-SEN':
                        entity_tokenlist = [token[0]]
                        entity_start = token[1][0]
                        entity_end = token[1][1]
                    else:
                        sen_token_startid = i
                        while i < len(doc_labels) - 1 and doc_labels[i] != 'E-SEN':
                            i += 1
                        sen_token_endid = i
                        entity_tokenlist = get_sequencelist(tokenlist, sen_token_startid, sen_token_endid)
                        entity_start = tokenlist[sen_token_startid][1][0]
                        entity_end = tokenlist[sen_token_endid][1][1]
                    doc_entities.append(Sentence(entity_tokenlist, entity_start, entity_end))
            # start of a list detected
            elif label in ['B-IT', 'S-IT']:
                # compute the start of the list
                li_token_startid = i
                while (li_token_startid > 0 and doc_labels[li_token_startid] != 'B-SEN'):
                    li_token_startid -= 1
                # compute the end of the list
                li_token_endid = i
                while li_token_endid + 1 < len(doc_labels) and doc_labels[li_token_endid + 1] not in ['B-SEN', 'S-SEN']:
                    li_token_endid += 1
                intro_tokens = []
                if li_token_startid < i:
                    intro_tokens = get_sequencelist(tokenlist, li_token_startid, i-1)
                entity_tokenlist = intro_tokens[:]
                entity_start = tokenlist[li_token_startid][1][0]
                entity_end = tokenlist[li_token_endid][1][1]
                #compute all items of a list
                itemlist = []
                item_token_startid = i
                item_token_endid = i
                while (item_token_endid <= li_token_endid):
                    label = doc_labels[item_token_endid]
                    if label =='E-IT':
                        item_tokenlist = get_sequencelist(tokenlist, item_token_startid, item_token_endid)
                        entity_tokenlist.extend(item_tokenlist)
                        bulletpoint = get_bulletpoint(item_token_startid, item_token_endid, tokenlist)
                        itemlist.append(
                            Item(item_tokenlist, tokenlist[item_token_startid][1][0], tokenlist[item_token_endid][1][1], bulletpoint))
                        item_token_startid = item_token_endid + 1
                    item_token_endid += 1
                doc_entities.append(List(entity_tokenlist, intro_tokens, entity_start, entity_end, itemlist))
                for item in itemlist:
                    doc_entities.append(item)
                i = li_token_endid
            i += 1
        entities.append(doc_entities)
    return entities

ROMAN_NUMERALS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
ROMAN_NUMERALS_CAPITAL = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

# counter to evaluate if the bulletlist is a roman numeral
def get_next_bullet(bulletlist, counter):
    """
    computes the next bulletpoint / list of tokens representing the next bulletpoint to a given bulletpoint

    :param bulletlist: list of tokens representing a bulletpoint
    :param counter: counter is used to determine if the bulletpoint is a possible roman numeral
    :return: a list of tokens representing the next bulletpoint to the given bulletpoint
    """
    if bulletlist is None or len(bulletlist) == 0:
        return []
    if len(bulletlist) == 1 and not bulletlist[0].isascii() or bulletlist[0] in ['~', '-', '+', '#', '$', '*']:
        return bulletlist
    lasttoken = ''
    lasttokenid = 0
    # compute the last alphanumeric token in the bulletlist, this token will be changed
    for i in range(len(bulletlist)):
        token = bulletlist[i]
        if token.isalpha() or token.isnumeric():
            lasttoken = token
            lasttokenid = i

    # if it is a numeric token just increment it
    if lasttoken.isnumeric():
        bulletlist[lasttokenid] = int(lasttoken) + 1
    elif lasttoken.isalpha():
        # to distinguish between a roman numeral and a sequence of letters, the counter is used to distinguish
        # for example, between v) following u) vs v) following iv)
        if counter < 10 and (ROMAN_NUMERALS[counter] == bulletlist[lasttokenid] or ROMAN_NUMERALS_CAPITAL[counter] == bulletlist[lasttokenid]):
            if counter < 10:
                if bulletlist[lasttokenid].isupper():
                    bulletlist[lasttokenid] = ROMAN_NUMERALS_CAPITAL[counter+1]
                else:
                    bulletlist[lasttokenid] = ROMAN_NUMERALS[counter + 1]
        # if it is a alphabetic token just get the next char
        elif len(lasttoken) == 1:
            bulletlist[lasttokenid] = chr(ord(lasttoken) + 1)
        else:
            bulletlist[lasttokenid] = lasttoken[:len(lasttoken) - 1] + chr(ord(lasttoken[len(lasttoken) - 1:]) + 1)
    return bulletlist

def get_hierarchy(entity):
    """
    :param entity: of type item1, item2, item3
    :return: return the hierarchy of an item
    """
    if isinstance(entity, Item1): return 1
    elif isinstance(entity, Item2): return 2
    elif isinstance(entity, Item3): return 3
    else: return None

def compute_hierarchy(y):
    """
    computes the hierarchy of the items of the lists in the documents in 4 steps:
    1. Step: compute the hierarchy for all items in the itemlist of a list
    2. Step: create the entities based on the computed hierarchy without the children
    3. Step: compute the children for each item type
    4. Step: compute the children for the list


    :param y: List of lists of entities of the type sentence, list, item - each list represents a document
    :return: List of lists of entities of the type sentence, list, item1, itmem2, item3 - each list represents a document
    """
    for doc_id in range(len(y)):
        doc = y[doc_id]
        for list_index in range(len(doc)):
            entity = doc[list_index]
            # the sentences are already constructed, so we only need to update the list and items
            if isinstance(entity, List):
                num_children = len(entity.children)
                # 1. Step: compute the hierarchy for all items in the itemlist of a list
                # tmp_itemlist stores all items with the computed hierarchy in the format [(hierachy, item],...]
                # bulletdict stores the actual bullet of the hierarchies
                tmp_itemlist = []
                bulletdict = {}
                hierarchy = 1
                for i in range(len(entity.children)):
                    item = entity.children[i]
                    # the first elemten of entity.items is always on hierarchy level 1 -> item1
                    if i == 0:
                        bulletdict[1] = (item.bulletlist,0)
                        tmp_itemlist.append((hierarchy, item))
                    else:
                        tmp_hierarchy = hierarchy
                        # check if the item fits in one of the already existing hierarchy levels
                        while (1 <= tmp_hierarchy <= 3):
                            (bulletlist, counter) = bulletdict[tmp_hierarchy]
                            if item.bulletlist == get_next_bullet(bulletlist[:], counter):
                                hierarchy = tmp_hierarchy
                                bulletdict[hierarchy] = (item.bulletlist, counter + 1)
                                break
                            tmp_hierarchy -= 1
                        # if it fits in no hierarchy level (so tmp_hierarchy == 0) the item is on the next hierarchy
                        if tmp_hierarchy == 0 and hierarchy <= 3:
                            if hierarchy < 3:
                                hierarchy += 1
                            bulletdict[hierarchy] = (item.bulletlist,0)
                        tmp_itemlist.append((hierarchy, item))
                # 2. Step: create the entities based on the computed hierarchy without the children
                itemlist = []
                for i in range(len(tmp_itemlist)):
                    (hierarchy, item) = tmp_itemlist[i]
                    tokenlist = item.tokenlist
                    # startid and endid are the relative ids in the doc text string
                    startid = item.startid
                    endid = item.endid
                    children = []
                    bulletlist = item.bulletlist
                    # compute the endid of the last child-item of the item if there are child-items
                    j = i + 1
                    while j < len(tmp_itemlist) and tmp_itemlist[j][0] > hierarchy:
                        (tmp_hierarchy, tmpitem) = tmp_itemlist[j]
                        tokenlist.extend(tmpitem.tokenlist)
                        endid = tmpitem.endid
                        j += 1
                    if hierarchy == 1:
                        itemlist.append(Item1(tokenlist, startid, endid, children,bulletlist))
                    elif hierarchy == 2:
                        itemlist.append(Item2(tokenlist, startid, endid, children, bulletlist))
                    elif hierarchy == 3:
                        itemlist.append(Item3(tokenlist, startid, endid, bulletlist))
                # 3. Step compute Children of Item1 - Item2, Item3 cant have a child by the definition
                for i in [2,1]:
                    for j in range(len(itemlist)):
                        if get_hierarchy(itemlist[j]) == i:
                            h = j + 1
                            hierarchy = get_hierarchy(itemlist[j])
                            while h < len(itemlist) and hierarchy < get_hierarchy(itemlist[h]):
                                if hierarchy + 1 == get_hierarchy(itemlist[h]):
                                    itemlist[j].children.append(itemlist[h])
                                h += 1
                # 4. Step compute children for List element
                list_children = []
                for item in itemlist:
                    if get_hierarchy(item) == 1:
                        list_children.append(item)
                # add all computed items to y
                entity.children = list_children
                doc[list_index] = entity
                doc = doc[:list_index + 1] + itemlist + doc[list_index + num_children + 1:]
            y[doc_id] = doc
    return y






