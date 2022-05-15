import json
from nltk.tokenize import RegexpTokenizer

class DatasetCreator:
    def __create_tags(self, tokens, is_sen):
        """
        computes a tag/label list tags/labels to a given sequence of tokens

        :param tokens: list of tokens
        :param is_sen: indicates if the sequence of tokens represents a sentence or a list

        :return:
        tags: list of tags/labels (strings) for a given sequence of tokens
        """
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

    def create_dataset(self, pathlist):
        """
        entities: e_1 := sentence; e_2 := list; e_3 = item1; e_4 = item2; e_5 = item3

        :param pathlist: list of paths to the training data to read

        :return:
        X_train: List of lists of tuples of (text, tokens_with_span); each list representing a document
            text: string of the document text
            tokens_with_span: a list of tuples of the structure (token, (startid, endid)) where the ids are relative to text

        y_train: list of lists of string
            Labels for several documents, each list entry represents one document
        """
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
            while(i < len(entities)):
                entity = entities[i]
                entity_text = entity["offsets"][0]["text"]
                entity_startid = entity["offsets"][0]["start"]
                # tokenize the text from the entity
                entity_tokens = tokenizer.tokenize(entity_text)
                entity_spans = list(tokenizer.span_tokenize(entity_text))

                # sentence
                if entities[i]["classId"] == "e_1":
                    tags.extend(self.__create_tags(entity_tokens, True))

                # get only the introductory listtext without any items, so only the text until the start of the first item
                elif entities[i]["classId"] == "e_2":
                    entity_end = (((entities[i+1])["offsets"])[0])["start"] - entity_startid
                    while (entity_end > 0 and entity_text[entity_end - 1] in [' ', '\n']):
                        entity_end -= 1
                    entity_text = entity_text[0:entity_end]
                    entity_tokens = tokenizer.tokenize(entity_text)
                    entity_spans = list(tokenizer.span_tokenize(entity_text))
                    tags.extend(self.__create_tags(entity_tokens, True))

                # if item1 has children of item2, get only the introductory item1 without any children
                elif entities[i]["classId"] == "e_3" and i + 1 < len(entities) and entities[i + 1]["classId"] == "e_4":
                    entity_end = (((entities[i+1])["offsets"])[0])["start"] - entity_startid
                    while (entity_end > 0 and entity_text[entity_end - 1] in [' ', '\n']):
                        entity_end -= 1
                    entity_text = entity_text[0:entity_end]
                    entity_tokens = tokenizer.tokenize(entity_text)
                    entity_spans = list(tokenizer.span_tokenize(entity_text))
                    tags.extend(self.__create_tags(entity_tokens, False))

                # if item2 has children of > item3, get only the introductory item2 without any items of the class item3
                elif entities[i]["classId"] == "e_4" and i + 1 < len(entities) and entities[i + 1]["classId"] == "e_5":
                    entity_end = (((entities[i+1])["offsets"])[0])["start"] - entity_startid
                    while (entity_end > 0 and entity_text[entity_end - 1] in [' ', '\n']):
                        entity_end -= 1
                    entity_text = entity_text[0:entity_end]
                    entity_tokens = tokenizer.tokenize(entity_text)
                    entity_spans = list(tokenizer.span_tokenize(entity_text))
                    tags.extend(self.__create_tags(entity_tokens, False))

                # all items without children
                else:
                    tags.extend(self.__create_tags(entity_tokens, False))

                # adjust indexes relative to text
                for h in range(len(entity_tokens)):
                    entity_spans[h] = (entity_spans[h][0] + entity_startid, (entity_spans[h][1] + entity_startid))
                    entity_tokens[h] = (entity_tokens[h], entity_spans[h])
                tokens.extend(entity_tokens)
                i += 1

            X_train.append((text, tokens))
            y_train.append(tags)
            print("Created " + str(doc_id) + " of " + str(len(pathlist)) + " subsets")
            doc_id += 1
        return X_train, y_train
