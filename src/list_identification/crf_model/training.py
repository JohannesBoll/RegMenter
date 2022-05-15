import joblib
from sklearn.pipeline import Pipeline
from sklearn_crfsuite import CRF
from create_dataset_tagtog import DatasetCreator
from crf_transformer import CRFTransformer

def training(training_pathlist):
    creator = DatasetCreator()
    X, y = creator.create_dataset(pathlist)
    # set your windowssize for the transformer, we use a windowssize of 3
    transformer = CRFTransformer(3)
    # set all hyperparameters
    crf = CRF(
        verbose=True,
        all_possible_states=False,
        all_possible_transitions=True,
        num_memories=6,
        period=10,
        min_freq=0,
        c1=1.5376818844585831,
        c2=0.02523805308490959,
        epsilon=1e-5,
        delta=1e-5,
        max_linesearch=20,
        linesearch='MoreThuente',
        max_iterations = 900
    )
    pipeline = Pipeline([
        ('feature_creator', transformer),
        ('crf_clf', crf)
    ])
    pipeline.fit(X,y)
    joblib.dump(pipeline, 'pipeline.pkl', compress = 1)

if __name__ == "__main__":
    # set the pathlist for the training data
    pathlist = [
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/Train/CANDRIAM_GF_LU1220230442_2015_K_P_R_A.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/Train/Credit_Suisse_Fund_I_(Lux)_2012_X_P_X_X.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/Train/Credit_Suisse_Fund_I_(Lux)_2012_X_P_X_X-V2.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/Train/EdR_Private_Equity_Select_Access_Fund_S.A._SICAV-SIF-Amethis_II__Sub-Fund_2018_K_X_X_X.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/Train/Dexia_Equities_L_2011_X_P_X_X.finsbd2.json',
        '/Users/johannesboll/Desktop/Studium/Semester 9/Bachelorarbeit/NLP BA/Bachelor-Thesis/Implementation/data/customized Data/Train/Invesco_Funds_SICAV_2013_X_P_X_A.finsbd2.json',
    ]
    training(pathlist)