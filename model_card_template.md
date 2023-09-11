# Model Card
## Model Details
Tran Minh Duc created this in Sep 8th. A SVM Classifier using default hyper-parameter of scikit-learn
## Intended Use
The model is used to predict personal income whether over 50000 USD based on personal information. 
## Training Data
The data is collected from [here](https://archive.ics.uci.edu/dataset/20/census+income). Extraction was done by Barry Becker from the 1994 Census database.  A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person makes over 50K a year.
## Evaluation Data
The evaluation data is taken randomly from 20% of the training data.
## Metrics
**Precision**: 0.988

**Recall**: 0.15833333333333333 

**F-Beta**: 0.2729281767955801

## Ethical Considerations
The dataset introduces bias for men when the proportion of high earners in men is much higher in women. In addition, there is a large disparity in the proportion of high earners across different races.
## Caveats and Recommendations
This model has high precision but low recall so it is better to only consider the positive predictions.