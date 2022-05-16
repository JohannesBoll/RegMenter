# RegMenter
The RegMenter (Regulatory Document Segmenter) is the proof-of-concept implementation of the Bachelor's Thesis "Identification and Complementation of
List Structures in Regulatory Documents" written at the chair for Information Systems and Business Process Management of the TUM.
Sentence boundary detection is a central component of any natural language processing application. While most research explores text segmentation as a general task, conventional sentence boundary detection systems show weaknesses in segmenting documents with domain specific formal and structural properties, such as regulatory documents. The root cause for these performance issues in processing such texts is the occurrence of complex structures, like lists or enumerations.
With RegMenter we offer a modular SBD system, which focuses on the identification and completion of lists in regulatory documents.

## Components
The RegMenter is a fully modular SBD System consisting of the following components:
1. `list_idenftification`: identifies sentences, lists and list items
   - `list_identification/parser`: parse PDF documents to plaintext
   - `list_identification/tokenizer`: tokenize plain text into token based on our tokenization strategy
   - `list_identification/crf_model`: sequence model to detect sentence and list boundaries, consisting of:
     - `list_identification/create_dataset_tagtog.py`: used for training to load annotated data
     - `list_identification/crf_transformer`: transformer for feature extraction
     - `list_identification/training.py`: can be used for training, you can add own data or change hyperparameters
     - `list_identification/pipeline.pkl`: pre-trained model, use `joblib` to load
   - `list_identification/lstm_models`: alternative sequence models, still under construction and not working so far
   - `list_identification/postprecessing`: segments labels into sentences, list and list items and identifies the hierarchy of the list and its items
2. `list_complementation`: used to complement the identified items into complete sentences using the `grammar_checker`. If a sentence is not complementable the user is notified. Finally output all detected sentences from `list_idenftification` together with the detected sentences of the complementation.

The final output of our SBD Systems creates for each PDF document a list of string (token) lists, where each string list represents a sentence.

## Install

```
git clone https://github.com/JohannesBoll/RegMenter
cd RegMenter/src
pip3 install -r requirements.txt
python -m spacy download en_core_web_sm
```
## Usage:
In the following we add some details about the usage of the RegMenter: 
### Training:
The CRF sequence model is the central element of our approach. To train it with own data or other hyperparameters the script `src/list_identification/crf_model/training.py` can be used. For this, the paths to the training data must be adapted in `training.py`.
We provide an already pre-trained model with the hyperparameters that were also used in the thesis. The hyperparameters used for training of the pre-trained model are the default values of `src/list_identification/crf_model/training.py`. The pre-trained model can be found at: `src/list_identification/crf_model/pipeline.pkl`.

### Prediction:
To segment PDF docs into sentences by our SBD system `src/predict.py` can be used. Before using it, the path of the location of the pre-trained model in `src/predict.py` must be adjusted. Since the list complementation module also uses the sequence model to segment the individual list items, the path to the location of the pre-trained model must also be adjusted in `src/list_identification/list_complementation/complementation/py` before use.
Before starting the prediction, the paths to the PDFs has also be adjusted in `predict.py`.

### Evaluation:
To reproduce the evaluation from the thesis the script `src/evaluate.py` can be used. Before using it, the path of the location of the pre-trained model in `src/predict.py` must be adjusted. Since the list complementation module also uses the sequence model to segment the individual list items, the path to the location of the pre-trained model must also be adjusted in `src/list_identification/list_complementation/complementation/py` before use. Finally the pathlist containing all paths to the annotated test data have to be adjusted in `src/evaluate.py`.