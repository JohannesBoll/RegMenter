from create_dataset_tagtog import DatasetCreator
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.initializers import Constant
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn_crfsuite import metrics

def createData(X, y, windowsize):
    """
    segment data into segments of windowssize with a hop size to get overlapping segments
    used since LSTM can only handle fixed input size
    """
    X_new, y_new = [], []
    hop  = 50
    for i in range(len(X)):
        X_tmp = X[i]
        y_tmp = y[i]
        X_chunks = [X_tmp[i:i+windowsize] for i in range(0, len(X_tmp), hop)]
        y_chunks = [y_tmp[i:i+windowsize] for i in range(0, len(y_tmp), hop)]
        X_new.extend(X_chunks)
        y_new.extend(y_chunks)
    return X_new, y_new

def main():
    pathlist = [
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/CANDRIAM_GF_LU1220230442_2015_K_P_R_A.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/Credit_Suisse_Fund_I_(Lux)_2012_X_P_X_X.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/Credit_Suisse_Fund_I_(Lux)_2012_X_P_X_X-V2.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/EdR_Private_Equity_Select_Access_Fund_S.A._SICAV-SIF-Amethis_II__Sub-Fund_2018_K_X_X_X.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/Dexia_Equities_L_2011_X_P_X_X.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/Invesco_Funds_SICAV_2013_X_P_X_A.finsbd2.json']
    test_pathlist = [
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/Test/Arabesque_SICAV_LU1023698662_2016_X_P_X_A.pdf.ann.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/Test/MAGALLANES_VALUE_INVESTORS_UCITS_LU1330191542_2016_X_P_X_X.pdf.ann.json']

    #create X, y and X_test, y_test
    creator = DatasetCreator()
    X, y = creator.create_dataset(pathlist)
    X_test, y_test = creator.create_dataset(test_pathlist)

    #include GloVe and create dict for tokens
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(X)
    tokenizer.fit_on_texts(X_test)
    X = tokenizer.texts_to_sequences(X)
    num_words = len(tokenizer.word_index) + 1
    embedding_dict = {}
    f = open('glove.840B.300d.txt')
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        embedding_dict[word] = coef

    #dim of word vectors
    embedding_dim = 300

    # first create a matrix of zeros, this is our embedding matrix
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # for each word in out tokenizer lets try to find that work in our w2v model
    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            # we found the word - add that words vector to the matrix
            embedding_matrix[i] = embedding_vector
        else:
            # doesn't exist, assign a nullvector
            embedding_matrix[i] = np.zeros(embedding_dim)

    windowsize = 300

    # segment data into fixed sized segments (see windowssize)
    X, y = createData(X, y, windowsize)

    tag2index = {t: i for i, t in enumerate(['-PAD-', 'B-IT', 'E-IT', 'B-SEN', 'E-SEN', 'S-SEN', 'O'])}
    y_train = []

    # map tag to index and pad eventually to short sequences to windowssize
    for doc in y:
        y_train.append([tag2index[t] for t in doc])
    X = pad_sequences(X, padding='post', maxlen=windowsize)
    y = pad_sequences(y_train,padding='post', maxlen=windowsize)

    X = array(X)
    X = X.reshape(len(X), len(X[0]))
    y = array(y)
    y = y.reshape(len(y), len(y[0]))

    #create testdata
    tokenizer.fit_on_texts(X_test)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test, y_test = createData(X_test, y_test, windowsize)
    X_test = pad_sequences(X_test, padding='post', maxlen=windowsize)
    X_test = array(X_test)
    X_test = X_test.reshape(len(X_test), len(X_test[0]))
    y_test_tmp = []
    for doc in y_test:
        y_test_tmp.append([tag2index[t] for t in doc])
    y_test = pad_sequences(y_test_tmp, padding='post', maxlen=windowsize)
    y_test = array(y_test)
    y_test = y_test.reshape(len(y_test), len(y_test[0]))

    model = Sequential()
    model.add(Embedding(num_words,
                    embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=windowsize,
                    trainable=False))
    model.add(LSTM(256, return_sequences=True, dropout=0.2))
    model.add(LSTM(128, return_sequences=True, dropout=0.1))
    model.add(Dense(len(tag2index)))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    model.summary()
    model.fit(X, y, epochs=4, batch_size=20, validation_split=0.1)
    y_pred = model.predict(X_test, verbose=1)

    def logits_to_tokens(sequences, index):
        token_sequences = []
        for categorical_sequence in sequences:
            token_sequence = []
            for categorical in categorical_sequence:
                token_sequence.append(index[np.argmax(categorical)])

            token_sequences.append(token_sequence)

        return token_sequences

    y_pred_test = logits_to_tokens(y_pred,{i: t for t, i in tag2index.items()})
    inv_map = {v: k for k, v in tag2index.items()}
    y_test_tmp = []
    for doc in y_test:
        tmp = []
        for h in range(len(doc)):
            tmp.append(inv_map[doc[h]])
        y_test_tmp.append(tmp)
    y_test = y_test_tmp
    m = MultiLabelBinarizer().fit(y_test)
    print(f1_score(m.transform(y_test), m.transform(y_pred_test), average='macro'))
    sorted_labels = sorted(
        ['O', '-PAD-', 'B-SEN', 'E-SEN'],
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred_test, labels=sorted_labels, digits=3
    ))


if __name__ == "__main__":
    main()
