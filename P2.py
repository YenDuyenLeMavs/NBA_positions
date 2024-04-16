import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

nba_stats_file = "nba_stats.csv"
nba_stats = pd.read_csv(nba_stats_file, header = 0)
columns_to_drop=['Pos','G', 'GS', 'Age', 'FTA', 'FT%', 'FT', '3P%', '2P%']
nba_data = nba_stats.drop(columns=columns_to_drop)
nba_target = nba_stats['Pos']

test_file = "dummy_test.csv"
test_records = pd.read_csv(test_file, header = 0)
test_feature = test_records.drop(columns=columns_to_drop)
if 'Predicted Pos' in test_feature.columns:
    test_feature = test_feature.drop(columns=['Predicted Pos'])
test_class = test_records['Pos']

train_feature, validation_feature, train_class, validation_class = train_test_split(
    nba_data, nba_target, test_size=0.2, stratify=nba_target, random_state=0)

mlp = MLPClassifier(hidden_layer_sizes=(25, 25, 25, 25, 25, 10), max_iter=5000, random_state=0)
mlp.fit(train_feature, train_class)

print("\n**** 80/20 train/validation split ****\n")
train_prediction = mlp.predict(train_feature)
train_accuracy = accuracy_score(train_class, train_prediction)
print("Train set accuracy: {:.3f}".format(train_accuracy))
print("Train set confusion matrix:")
print(pd.crosstab(train_class, train_prediction, rownames=['True'], colnames=['Predicted'], margins=True))
print("")
validation_prediction = mlp.predict(validation_feature)
validation_accuracy = accuracy_score(validation_class, validation_prediction)
print("Validation set accuracy: {:.3f}".format(validation_accuracy))
print("Validation set confusion matrix:")
print(pd.crosstab(validation_class, validation_prediction, rownames=['True'], colnames=['Predicted'], margins=True))

print("\n**** Test model on test set ****\n")
mlp.fit(nba_data, nba_target)
test_prediction = mlp.predict(test_feature)
test_accuracy = accuracy_score(test_class, test_prediction)
print("Test set accuracy: {:.3f}".format(test_accuracy))
print("Test set confusion matrix:")
print(pd.crosstab(test_class, test_prediction, rownames=['True'], colnames=['Predicted'], margins=True))

print("\n**** 10-fold stratified cross-validation ****\n")
scores = cross_val_score(mlp, nba_data, nba_target, cv=10)
print("Cross-validation accuracy for each fold: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
print("")