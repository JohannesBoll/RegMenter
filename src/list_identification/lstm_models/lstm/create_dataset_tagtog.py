import json
import random
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import DictVectorizer

class DatasetCreator:
    # pathlist hier an Objekt binden oder nicht? TODO
    # def __init__(self):

    # creates the feature dict of a given token
    def __feature(self, tokenseq, tokenId, text, startid, endid):
        special_characters = "#$%&'*+/<=>@[\]^_`{|}~"
        token = tokenseq[tokenId]
        newlinesbefore, newlinesafter, spacesbefore, spacesafter, snbefore, snafter = 0,0,0,0,0,0
        i1, i2, i3, i4, i5, i6 = startid - 1, endid, startid - 1, endid, startid - 1, endid
        while i1 > 0 and text[i1] == '\n':
            newlinesbefore += 1
            i1 -= 1
        while i2 < len(text) and text[i2] == '\n':
            newlinesafter += 1
            i2 += 1
        while i3 > 0 and text[i3] == ' ':
            spacesbefore += 1
            i3 -= 1
        while i4 < len(text) and text[i4] == ' ':
            spacesafter += 1
            i4 += 1
        while i5 > 0 and text[i5] in [' ', '\n']:
            snbefore += 1
            i5 -= 1
        while i6 < len(text) and text[i6] in [' ', '\n']:
            snafter += 1
            i6 += 1
        return {
            #basic features about the main token
            'length': len(token),
            'is_ascii': token.isascii(),
            'is_numeric': token.isdigit(),
            'is_capitalized': token[0].isupper(),
            'sentence_terminating_char': token in ['!', '?', '.'],
            'doppelpunkt': token == ':',
            'comma': token == ',',
            'semicolon': token == ';',
            'punctuation': token == ':',
            'parentheses': token == '\"',
            'brackets': token in ['(', ')'],
            'is_special': True if (any(c in special_characters for c in token)) else False,
            'undefined_unicode': token == '',
            '–' : token == '–',
            '-' : token == '-',
            '•': token == '•',
            '�': token == '�',
            'undef': tokenseq[tokenId] == '\uf0a7',

            #features from the sorrounding context
            'spacebefore': True if spacesbefore > 0 else False,
            'spacesbefore': spacesbefore,
            'spaceafter': True if spacesafter > 0 else False,
            'spacesafter': spacesafter,
            'newlinebefore': True if newlinesafter > 0 else False,
            'newlinesbefore': newlinesbefore,
            'newlineafter': True if newlinesafter > 0 else False,
            'newlinesafter': newlinesbefore,
            'snbefore': snbefore,
            'snafter': snafter,

            #cases like a) etc
            'possible_multichar_item_start': True if ((tokenId + 1 < len(tokenseq) and tokenseq[tokenId + 1] == ')')
                                            or (tokenId - 1 > 0 and tokenseq[tokenId - 1] == '(' and tokenId + 1 < len(tokenseq) and tokenseq[tokenId + 1] == ')')) else False
        }

    # get tags to a list of tokens (tokens), is_sen shows if the token-sequence is a sentence or list
    def __create_tags(self, tokens, is_sen):
        tags = []
        if is_sen:
            if len(tokens) == 1:
                return ['O']
            else:
                tags = ['O']
                tags.extend((len(tokens) - 2) * ['O'])
                tags.append('O')
        else:
            if len(tokens) == 1:
                return ['O']
            else:
                tags = ['B-LI']
                tags.extend((len(tokens) - 2) * ['O'])
                tags.append('E-LI')
        return tags

    def sub_lists(self, l):
        lists = [[]]
        for i in range(len(l) + 1):
            for j in range(i):
                lists.append(l[j: i])
        return lists

    def getStart(self, text, startid):
        if text[startid-1] == '\n':
            return startid - 1
        if text[startid-2] == '\n':
            return startid - 2
        return startid
    # create X_train and y_train
    # expects a list of file paths to the files to read
    # encoded entites: e_1 := sentence; e_2 := list
    def create_training_dataset(self, pathlist):
        sentences = []
        lists = []
        items = []
        #iterate over all given inputs/data
        for path in pathlist:
            with open(path) as json_file:
                data = json.load(json_file)
            # we need only the sentences and lists from the JSON for classification
            tokens = []
            tags = []
            entities_tmp = data['entities']
            entities = []
            for ent in entities_tmp:
                if ent["classId"] == "e_1":
                    sentences.append(ent)
                if ent["classId"] == "e_2":
                    lists.append(ent)
                if ent["classId"] == "e_3":
                    items.append(ent)
        items_seg = []
        #create for every list entry an entry with all items of the list in sub
        for li in lists:
            item_sublist = []
            liststartId = li["offsets"][0]["start"]
            listtext = li["offsets"][0]["text"]
            listendId = liststartId + len(listtext)
            for i in range(len(items)):
                item = items[i]
                itemstartId = item["offsets"][0]["start"]
                itemtext = item["offsets"][0]["text"]
                itemendId = itemstartId + len(itemtext)
                if itemstartId >= liststartId and itemendId <= listendId:
                    itemstartId = self.getStart(listtext, itemstartId - liststartId)
                    itemendId = itemendId-liststartId
                    text = listtext[itemstartId:itemendId]
                    item_sublist.append(text)
                if itemstartId > listendId:
                    break
            items_seg.append(item_sublist)
        newlists = []
        for (li, itemlist) in zip(lists, items_seg):
            sublists = self.sub_lists(itemlist)
            listtext = li["offsets"][0]["text"]
            for sublist in sublists:
                if len(sublist) < len(itemlist):
                    if len(sublist) == 0:
                        newlists.append(listtext)
                    else:
                        text = listtext
                        for itemtext in sublist:
                            text = text.replace(itemtext, '')
                        newlists.append(text)
        X_train, y_train = [], []
        print("Starting dataset creation...")
        docid = 1
        # create for each document in filelist tokens and tags
        tokenizer = RegexpTokenizer('[\w\'-]+|[^\w\s]+')
        v = DictVectorizer(sparse=False)
        #(sentence1,list,sentence2) - ((tokens with span), labels, text)
        for li in newlists:
            sentence1text = random.choice(sentences)["offsets"][0]["text"]
            sentence2text = random.choice(sentences)["offsets"][0]["text"]
            tokenssen1 = tokenizer.tokenize(sentence1text)
            tagssen1 = self.__create_tags(tokenssen1, True)
            tokenslist = tokenizer.tokenize(li)
            tagslist = self.__create_tags(tokenslist, False)
            tokenssen2 = tokenizer.tokenize(sentence2text)
            tagssen2 = self.__create_tags(tokenssen2, True)
            tags = tagssen1 + tagslist + tagssen2
            text = sentence1text + ' ' + li + ' ' + sentence2text
            tokens = tokenizer.tokenize(text)
            spans = list(tokenizer.span_tokenize(text))
            featurelist = []
            tokenid = 0
            for token, span in zip (tokens, spans):
                feature_dict = self.__feature(tokens, tokenid, text, span[0], span[1])
                tokenid += 1
                featurelist.extend(v.fit_transform(feature_dict))
            X_train.append(featurelist)
            y_train.append(tags)


        return X_train, y_train


    def create_data_from_text(self, text):
        tokenizer = RegexpTokenizer('[\w\'-]+|[^\w\s]+')
        v = DictVectorizer(sparse=False)
        tokens = tokenizer.tokenize(text)
        spans = list(tokenizer.span_tokenize(text))
        X = []
        for token, span in zip(tokens, spans):
            feature_dict = self.__feature(tokens, tokenid, text, span[0], span[1])
            tokenid += 1
            featurelist.extend(v.fit_transform(feature_dict))
        X.append(featurelist)

