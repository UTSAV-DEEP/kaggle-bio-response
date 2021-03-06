# kaggle-bio-response


##Pre-requisite

- Login to [kaggle](https://www.kaggle.com) and download [train.csv](https://www.kaggle.com/c/bioresponse/download/train.csv).
Put this downloaded file in the folder:
<br>
<b>kaggle-bio-response/data/raw</b>

- Install python3
<br>
Install anaconda for python3 and create anaconda python environment and install (or update) the required python libraries given in environment.yml.
<br>
install: <code>conda env create -f environment.yml -n kaggle</code>
<br>
update: <code>conda env update -f environment.yml -n kaggle</code>
<br>
environment.yml contains following dependencies that are required in this project:
<pre>
dependencies:
 - python=3.5
 - numpy
 - scipy
 - scikit-learn
 - jupyter
 - pandas
 - seaborn
 - click
 - keras
 - pip:
   - watermark
</pre>

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
│   └── execution_report.ipynb
├── README.md
├── reports
│   ├── capstone project report.docx
│   ├── capstone project report.pdf
│   ├── execution_report.html
│   ├── figures
│   │   ├── evaluation
│   │   │   ├── benchmark_roc_curve.png
│   │   │   └── roc_curves.png
│   │   ├── processed_target_corr_plot.png
│   │   └── raw_target_corr_plot.png
│   └── model_performance.txt
└── src
    ├── commons
    │   ├── constants.py
    │   ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   ├── preprocess.py
    ├── evaluation
    │   ├── __init__.py
    │   ├── model_performance.py
    ├── __init__.py
    ├── models
    │   ├── __init__.py
    │   ├── keras_nn.py
    │   ├── logistic_regression.py
    │   ├── random_forest.py
    │   ├── svm.py
    │   └── train_models.py
    └── visualization
        ├── exploratory.py
        ├── __init__.py

</pre>



## Running the project

Run these python files in the sequence:

- src/data/preprocess.py
- src/visualization/exploratory.py
- src/models/train_models.py
- src/evaluation/model_performance.py


## Perfromance metrics

<pre>
Calculating benchmark model performance:
_____________________________________________________________________________________

Performance of SVM
-------------------------------------------------------------------------------------

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


accuracy_score:	0.744

roc_auc_score:	0.743

log_loss:	0.524
_____________________________________________________________________________________

Calculating improved models performances:
_____________________________________________________________________________________

Performance of RandomForest
-------------------------------------------------------------------------------------

confusion_matrix:
[[279  65]
 [ 72 335]]

classification_report:
              precision    recall  f1-score   support

           0       0.79      0.81      0.80       344
           1       0.84      0.82      0.83       407

   micro avg       0.82      0.82      0.82       751
   macro avg       0.82      0.82      0.82       751
weighted avg       0.82      0.82      0.82       751


accuracy_score:	0.818

roc_auc_score:	0.817

log_loss:	0.438
_____________________________________________________________________________________
_____________________________________________________________________________________

Performance of LogisticRegression
-------------------------------------------------------------------------------------

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


accuracy_score:	0.727

roc_auc_score:	0.726

log_loss:	0.703
_____________________________________________________________________________________
_____________________________________________________________________________________

Performance of SVM
-------------------------------------------------------------------------------------

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


accuracy_score:	0.788

roc_auc_score:	0.786

log_loss:	0.461
_____________________________________________________________________________________
_____________________________________________________________________________________

Performance of KerasNN
-------------------------------------------------------------------------------------

confusion_matrix:
[[261  83]
 [ 79 328]]

classification_report:
              precision    recall  f1-score   support

           0       0.77      0.76      0.76       344
           1       0.80      0.81      0.80       407

   micro avg       0.78      0.78      0.78       751
   macro avg       0.78      0.78      0.78       751
weighted avg       0.78      0.78      0.78       751


accuracy_score:	0.784

roc_auc_score:	0.782

log_loss:	0.844
_____________________________________________________________________________________

</pre>




