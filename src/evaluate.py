import joblib

from list_identification.crf_model.create_dataset_tagtog import DatasetCreator
from list_identification.postprocessing.evaluate import evaluate_labels, evaluate_entites
from list_identification.postprocessing.SBDPostprocessing import label_completion, segment_tokens, compute_hierarchy
from list_complementation.evaluate import create_ref, create_pred_data, create_true_data, evaluate_complementation

def evaluate(pathlist_identification, pathlist_complementation):
    """
    List Identification Task Evaluation:
    """
    # start with the evaluation of the list detection module
    # create the tokenlist (X), and the test label set containg all true labels
    creator = DatasetCreator()
    X, y_true = creator.create_dataset(pathlist_identification)
    # load the sequence model, the default sequence model is trained with the hyperparameters from the thesis
    sequence_model = joblib.load(# TODO)
    y_pred = sequence_model.predict(X)
    # evaluate before error correction / label completion
    print("Performance before Error Corrcetion:")
    evaluate_labels(y_true, y_pred)
    print("Performance after Error Corrcetion:")
    y_pred = label_completion(y_pred)
    # evaluate after error correction / label completion
    evaluate_labels(y_true, y_pred)

    # do segmentation and hierarchy detection
    y_pred = segment_tokens(X, y_pred)
    y_pred = compute_hierarchy(y_pred)

    # evaluate all created entites
    evaluate_entites(pathlist_identification, y_pred)

    """
    List Complementation Task Evaluation:
    """
    # evaluate based on data from the test set
    print("\nPerformance on Dataset:")
    y_true = create_true_data(pathlist_complementation)
    y_ref = create_ref(pathlist_complementation)
    evaluate_complementation(y_true, y_ref)
    # evaluate based on the prediciton
    print("\nPerformance on Predictions:")
    y_pred = create_pred_data(y_pred)
    evaluate_complementation(y_true, y_pred)

if __name__ == "__main__":
    # add pathlist for the list identification task
    pathlist_identification = [
        # Testing with annotated data
        # TODO
    ]
    # add pathlist for the list complementation task
    pathlist_complementation = [
        # Testing with annotated data
        # TODO
    ]
    evaluate(pathlist_identification, pathlist_complementation)

