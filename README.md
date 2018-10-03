# kaggle-bio-response

##Pre-requisite

- Login to [kaggle](https://www.kaggle.com) and download [train.csv](https://www.kaggle.com/c/bioresponse/download/train.csv).
Put this downloaded file in the folder:
<br>
<b>kaggle-bio-response/data/raw</b>

- Install python3
<br>
Install anaconda for python3 and create anaconda python environment and install (or update) the required python libraries.
<br>
install: <code>conda env create -f environment.yml -n kaggle</code>
<br>
update: <code>conda env update -f environment.yml -n kaggle</code>

- Install Pycharm IDE and open the cloned project in Pycharm.


## Project Structure

Here is the project structure that also shows the processed data file, generated models, plots, etc:
<pre>
.
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   │   └── processed.pickle
│   └── raw
│       ├── raw.pickle
│       └── train.csv
├── environment.yml
├── models
│   ├── benchmark
│   │   └── SVM.model
│   ├── KerasNN.h5
│   ├── KerasNN.yaml
│   ├── LogisticRegression.model
│   ├── RandomForest.model
│   └── SVM.model
├── notebooks
│   └── 00-initial-exploration.ipynb
├── README.md
├── reports
│   └── figures
│       └── evaluation
│           ├── benchmark_roc_curve.png
│           └── roc_curves.png
└── src
    ├── commons
    │   ├── constants.py
    │   ├── __init__.py
    │   └── __pycache__
    │       ├── constants.cpython-35.pyc
    │       ├── constants.cpython-36.pyc
    │       ├── __init__.cpython-35.pyc
    │       └── __init__.cpython-36.pyc
    ├── data
    │   ├── __init__.py
    │   ├── preprocess.py
    │   └── __pycache__
    │       ├── __init__.cpython-35.pyc
    │       └── preprocess.cpython-35.pyc
    ├── evaluation
    │   ├── __init__.py
    │   └── model_performance.py
    ├── __init__.py
    ├── models
    │   ├── __init__.py
    │   ├── keras_nn.py
    │   ├── logistic_regression.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-35.pyc
    │   │   ├── keras_nn.cpython-35.pyc
    │   │   ├── logistic_regression.cpython-35.pyc
    │   │   ├── random_forest.cpython-35.pyc
    │   │   └── svm.cpython-35.pyc
    │   ├── random_forest.py
    │   ├── svm.py
    │   └── train_models.py
    ├── __pycache__
    │   ├── __init__.cpython-35.pyc
    │   └── __init__.cpython-36.pyc
    └── visualization
        ├── exploratory.py
        └── __init__.py
</pre>



## Running the project

Run these python files in the sequence:

- src/data/preprocess.py
- src/visualization/exploratory.py
- src/models/train_models.py
- src/evaluation/model_performance.py


## Perfromance metrics

<pre>
calculating benchmark model performance:

Performance of SVM
confusion_matrix:
[[251  93]
 [ 99 308]]
classification_report:
              precision    recall  f1-score   support

           0       0.72      0.73      0.72       344
           1       0.77      0.76      0.76       407

   micro avg       0.74      0.74      0.74       751
   macro avg       0.74      0.74      0.74       751
weighted avg       0.74      0.74      0.74       751

accuracy_score: 0.744
roc_auc_score: 0.743
log_loss: 8.830

Calculating other models performance:

Performance of RandomForest
confusion_matrix:
[[270  74]
 [ 69 338]]
classification_report:
              precision    recall  f1-score   support

           0       0.80      0.78      0.79       344
           1       0.82      0.83      0.83       407

   micro avg       0.81      0.81      0.81       751
   macro avg       0.81      0.81      0.81       751
weighted avg       0.81      0.81      0.81       751

accuracy_score: 0.810
roc_auc_score: 0.808
log_loss: 6.577

Performance of LogisticRegression
confusion_matrix:
[[244 100]
 [105 302]]
classification_report:
              precision    recall  f1-score   support

           0       0.70      0.71      0.70       344
           1       0.75      0.74      0.75       407

   micro avg       0.73      0.73      0.73       751
   macro avg       0.73      0.73      0.73       751
weighted avg       0.73      0.73      0.73       751

accuracy_score: 0.727
roc_auc_score: 0.726
log_loss: 9.428

Performance of SVM
confusion_matrix:
[[261  83]
 [ 76 331]]
classification_report:
              precision    recall  f1-score   support

           0       0.77      0.76      0.77       344
           1       0.80      0.81      0.81       407

   micro avg       0.79      0.79      0.79       751
   macro avg       0.79      0.79      0.79       751
weighted avg       0.79      0.79      0.79       751

accuracy_score: 0.788
roc_auc_score: 0.786
log_loss: 7.313

Performance of KerasNN
confusion_matrix:
[[258  86]
 [ 78 329]]
classification_report:
              precision    recall  f1-score   support

           0       0.77      0.75      0.76       344
           1       0.79      0.81      0.80       407

   micro avg       0.78      0.78      0.78       751
   macro avg       0.78      0.78      0.78       751
weighted avg       0.78      0.78      0.78       751

accuracy_score: 0.782
roc_auc_score: 0.779
log_loss: 7.543

</pre>




