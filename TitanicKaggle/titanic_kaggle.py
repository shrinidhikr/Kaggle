import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import linear_model, preprocessing, tree, model_selection, ensemble, grid_search
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

def write_prediction(prediction, name):
    PassengerId = np.array(test["PassengerId"]).astype(int)
    solution = pd.DataFrame(prediction, PassengerId, columns = ["Survived"])
    solution.to_csv(name, index_label = ["PassengerId"])


clean_data(train)
clean_data(test)


#--------Logistic Regression----------

target = train["Survived"].values
features = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

print("\nUse logistic regression")

logistic = linear_model.LogisticRegression()
logistic.fit(features, target)
print(logistic.score(features, target))

scores = model_selection.cross_val_score(logistic, features, target, scoring='accuracy', cv=10)
print(scores) 
print(scores.mean())

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
write_prediction(logistic.predict(test_features), "logistic_regression.csv")



#--------Polynomial Features----------

print("\nUse polynomial features")

poly = preprocessing.PolynomialFeatures(degree=2)
features_ = poly.fit_transform(features)

clf = linear_model.LogisticRegression(C=10)
clf.fit(features_, target)
print(clf.score(features_, target))

scores = model_selection.cross_val_score(clf, features_, target, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())

test_features_ = poly.fit_transform(test_features)
write_prediction(clf.predict(test_features_), "logistic_regression_poly.csv")


#--------Decision Tree----------

print("\nDecision Tree")

target = train["Survived"].values
features = train[["Pclass", "Sex", "Age", "Fare"]].values

decision_tree = tree.DecisionTreeClassifier(random_state = 1)
decision_tree = decision_tree.fit(features, target)

print(decision_tree.feature_importances_)
print(decision_tree.score(features, target))

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
prediction = decision_tree.predict(test_features)
write_prediction(prediction, "decision_tree.csv")

print("\nCorrect overfitting")

feature_names = ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]
features_two = train[feature_names].values
decision_tree_two = tree.DecisionTreeClassifier(
    max_depth = 7,
    min_samples_split = 2,
    random_state = 1)
decision_tree_two = decision_tree_two.fit(features_two, target)

print(decision_tree_two.feature_importances_)
print(decision_tree_two.score(features_two, target))
tree.export_graphviz(decision_tree_two, feature_names=feature_names, out_file="decision_tree_two.dot")

scores = model_selection.cross_val_score(decision_tree_two, features_two, target, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())

test_features_two = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
prediction_two = decision_tree_two.predict(test_features_two)
write_prediction(prediction_two, "decision_tree_two.csv")



#--------Random Forest----------

print("\nUse Random Forest classifier")

target = train["Survived"].values
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

forest = ensemble.RandomForestClassifier(
    max_depth = 7,
    min_samples_split = 4,
    n_estimators = 1000,
    random_state = 1,
    n_jobs = -1
)
forest = forest.fit(features_forest, target)

print(forest.feature_importances_)
print(forest.score(features_forest, target))

scores = model_selection.cross_val_score(forest, features_forest, target, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())

test_features_forest = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
prediction_forest = forest.predict(test_features_forest)
write_prediction(prediction_forest, "random_forest.csv")



#--------Gradient boosting classifier----------

print("\nUse gradient boosting classifier")

target = train["Survived"].values
features = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

gbm = ensemble.GradientBoostingClassifier(
    learning_rate = 0.005,
    min_samples_split=40,
    min_samples_leaf=1,
    max_features=2,
    max_depth=12,
    n_estimators=1500,
    subsample=0.75,
    random_state=1)
gbm = gbm.fit(features, target)

print(gbm.feature_importances_)
print(gbm.score(features, target))

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
prediction_gbm = gbm.predict(test_features)
write_prediction(prediction_gbm, "gbm.csv")



#--------KNN Classfier----------

print("\nKNN Classfier")

target = train["Survived"].values
features = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

knn = KNeighborsClassifier(n_neighbors=5)

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

KNNClassifier = knn.fit(features, target)
print(KNNClassifier.score(features, target))

prediction_knn = KNNClassifier.predict(test_features)
write_prediction(prediction_knn, "knn.csv")
