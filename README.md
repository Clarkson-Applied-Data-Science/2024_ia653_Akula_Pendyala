# Meeting-1 11-12-2024 4 PM
## Team Members:
- Durga Venkata Vamsi Akula - 0993203
- Dinesh Pendyala - 0999419

# Project Name: Toxic Comment Classification Challenge
The main aim of this project is to identify and classify toxic online comments. We intend to build a model that’s capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate.

Reference: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

# About Dataset
- The source of dataset is **Kaggle** website. According to Kaggle website, this dataset is a direct excerpt from Wikipedia's talk page edits. Through this project, we will hopefully help online discussion become more productive and respectful.
- There are 8 columns and **159571** records in training data. Out of 8 columns, 2 columns are simply ID, and the comment itself. The other 6 columns are **toxic**, **severe_toxic**, **obscene**, **threat**, **insult**, **identity_hate** which are labels.
![train_head](media/train_head.png)
- There are two csv files pertaining to training data. In total, there are **153164** records in test data. Out of these, ground truth for only **63978** test records are provided as these records are used for scoring in the competiton. Hence, we intend to use only these records as test data in this project.

![test_head](media/test_head.png)
![test_labels_head](media/test_labels_head.png)

# Plan of Action
- **Data Cleaning**: Text data always requires rigorous cleaning and should be formatted to fit our model's input requirements
- **Model selection**: Determining the appropriate models for this data set. We intend to implement four different models.
    1. Logistic Regression
    2. Multinomial Naive Bayes model
    3. RNNs model etc
- **Analysing results**: In Kaggle website, this competiton is scored based on accurcay and ROC & AUC scores. Hene, we decided to use accuracy and ROC&AUC scores as quality metrics.

# Questions
- Is the project big enough?
- How to handle some data where the comment is "?


