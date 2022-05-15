from create_dataset_tagtog import DatasetCreator
from lstm_transformer import LSTMTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, InputLayer, Dense, TimeDistributed, Activation, Masking, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf

# create data of fixed windowssize
def createData(X, y, windowsize):
    X_new, y_new = [], []
    for i in range(len(X)):
        X_tmp = X[i]
        y_tmp = y[i]
        X_chunks = [X_tmp[i:i+windowsize] for i in range(0, len(X_tmp), windowsize)]
        y_chunks = [y_tmp[i:i+windowsize] for i in range(0, len(y_tmp), windowsize)]
        X_new.extend(X_chunks)
        y_new.extend(y_chunks)
    return X_new, y_new

def main():
    creator = DatasetCreator()
    pathlist = [
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/CANDRIAM_GF_LU1220230442_2015_K_P_R_A.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/Credit_Suisse_Fund_I_(Lux)_2012_X_P_X_X.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/Credit_Suisse_Fund_I_(Lux)_2012_X_P_X_X-V2.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/EdR_Private_Equity_Select_Access_Fund_S.A._SICAV-SIF-Amethis_II__Sub-Fund_2018_K_X_X_X.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/Dexia_Equities_L_2011_X_P_X_X.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/train/Invesco_Funds_SICAV_2013_X_P_X_A.finsbd2.json'
        ]
    test_pathlist = [
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/Test/Arabesque_SICAV_LU1023698662_2016_X_P_X_A.pdf.ann.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/Test/MAGALLANES_VALUE_INVESTORS_UCITS_LU1330191542_2016_X_P_X_X.pdf.ann.json'
        ]
    X, y = creator.create_training_dataset(pathlist)

    # set windowssize for model
    windowsize = 300

    X, y = createData(X, y, windowsize)

    # transform labels\tags to index and pad sequences eventually
    tag2index = {t: i for i, t in enumerate(['-PAD-', 'B-LI', 'E-LI', 'O'])}
    tag2index['-PAD-'] = 0  # The special value used to padding
    y_train = []
    for doc in y:
        y_train.append([tag2index[t] for t in doc])
    X = pad_sequences(X, padding='post', maxlen=windowsize)
    y = pad_sequences(y_train,padding='post', maxlen=windowsize)
    y = y.reshape(len(y), len(y[0]))
    X_test, y_test = creator.create_training_dataset(test_pathlist)
    X = array(X)
    X = X.reshape(len(X), len(X[0]), len(X[0][0]))
    X_test, y_test = createData(X_test, y_test, windowsize)
    X_test = pad_sequences(X_test, padding='post', maxlen=windowsize)
    X_test = array(X_test)
    X_test = X_test.reshape(len(X_test), len(X_test[0]), len(X_test[0][0]))
    y_test_encoded = []
    for doc in y_test:
        y_test_encoded.append([tag2index[t] for t in doc])
    y_test_encoded = pad_sequences(y_test_encoded, padding='post', maxlen=windowsize)
    y_test_encoded = array(y_test_encoded)
    y_test_encoded = y_test_encoded.reshape(len(y_test_encoded), len(y_test_encoded[0]))
    # two layered LSTM model
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, dropout=0.3), input_shape=(windowsize, len(X[0][0])))
    model.add(LSTM(128, return_sequences=True, dropout=0.2))
    model.add(Dense(len(tag2index)))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])
    model.summary()
    model.fit(X, y, epochs=10, batch_size=20)
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
    model.evaluate(X_test, y_test_encoded)


if __name__ == "__main__":
    main()
