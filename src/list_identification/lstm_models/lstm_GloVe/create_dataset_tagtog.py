import json
from nltk.tokenize import RegexpTokenizer

class DatasetCreator:
    # pathlist hier an Objekt binden oder nicht? TODO
    # def __init__(self):

    # get tags to a list of tokens (tokens), is_sen shows if the token-sequence is a sentence or list
    def __create_tags(self, tokens, is_sen):
        tags = []
        if is_sen:
            if len(tokens) == 1:
                return ['S-SEN']
            else:
                tags = ['B-SEN']
                tags.extend((len(tokens) - 2) * ['O'])
                tags.append('E-SEN')
        else:
            if len(tokens) == 1:
                return ['S-IT']
            else:
                tags = ['B-IT']
                tags.extend((len(tokens) - 2) * ['O'])
                tags.append('E-IT')
        return tags

    # create X_train and y_train
    # expects a list of file paths to the files to read
    # encoded entites: e_1 := sentence; e_2 := list
    def create_dataset(self, pathlist):
        X_train, y_train = [], []
        # we use a simple tokenizer, based on RegEx, to split alphabetic, numerical and single special character
        tokenizer = RegexpTokenizer('[a-zA-Z-\']+|[0-9]+|[^a-zA-Z0-9\s]')
        print("Starting dataset creation...")
        doc_id = 1

        # create for each document in filelist tokens and tags
        for path in pathlist:
            with open(path) as json_file:
                data = json.load(json_file)
            tokens = []
            tags = []
            entities = data['entities']
            text = data['text']

            # entities: e_1 := sentence; e_2 := list; e_3 = item1; e_4 = item2; e_5 = item3
            i = 0
            while (i < len(entities)):
                entity = entities[i]
                entity_text = entity["offsets"][0]["text"]
                entity_startid = entity["offsets"][0]["start"]
                # tokenize the text from the entity
                entity_tokens = tokenizer.tokenize(entity_text)
                #entity_spans = list(tokenizer.span_tokenize(entity_text))

                # sentence
                if entities[i]["classId"] == "e_1":
                    tags.extend(self.__create_tags(entity_tokens, True))

                # get only the introductory listtext without any items, so only the text until the start of the first item
                elif entities[i]["classId"] == "e_2":
                    entity_end = (((entities[i + 1])["offsets"])[0])["start"] - entity_startid
                    while (entity_end > 0 and entity_text[entity_end - 1] in [' ', '\n']):
                        entity_end -= 1
                    entity_text = entity_text[0:entity_end]
                    entity_tokens = tokenizer.tokenize(entity_text)
                    #entity_spans = list(tokenizer.span_tokenize(entity_text))
                    tags.extend(self.__create_tags(entity_tokens, True))

                # if item1 has children of item2, get only the introductory item1 without any children
                elif entities[i]["classId"] == "e_3" and i + 1 < len(entities) and entities[i + 1]["classId"] == "e_4":
                    entity_end = (((entities[i + 1])["offsets"])[0])["start"] - entity_startid
                    while (entity_end > 0 and entity_text[entity_end - 1] in [' ', '\n']):
                        entity_end -= 1
                    entity_text = entity_text[0:entity_end]
                    entity_tokens = tokenizer.tokenize(entity_text)
                    #entity_spans = list(tokenizer.span_tokenize(entity_text))
                    tags.extend(self.__create_tags(entity_tokens, False))

                # if item2 has children of > item3, get only the introductory item2 without any items of the class item3
                elif entities[i]["classId"] == "e_4" and i + 1 < len(entities) and entities[i + 1]["classId"] == "e_5":
                    entity_end = (((entities[i + 1])["offsets"])[0])["start"] - entity_startid
                    while (entity_end > 0 and entity_text[entity_end - 1] in [' ', '\n']):
                        entity_end -= 1
                    entity_text = entity_text[0:entity_end]
                    entity_tokens = tokenizer.tokenize(entity_text)
                    #entity_spans = list(tokenizer.span_tokenize(entity_text))
                    tags.extend(self.__create_tags(entity_tokens, False))

                # all items without children
                else:
                    tags.extend(self.__create_tags(entity_tokens, False))

                tokens.extend(entity_tokens)
                i += 1

            X_train.append(tokens)
            y_train.append(tags)
            print("Created " + str(doc_id) + " of " + str(len(pathlist)) + " subsets")
            doc_id += 1
        return X_train, y_train