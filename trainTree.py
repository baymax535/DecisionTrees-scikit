from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt



#initialize

print("\nCalifornia: \n")
house = fetch_openml(data_id=43979)
print(house.details, "\n")
house.frame.info()
print(house.data)
print(house)
print(house.target)

#trainTree

myTree = DecisionTreeClassifier(criterion="entropy") #untrained decision tree with entropy criteria. default uses Gini index

myTree.fit(house.data, house.target) #trained

#print(tree.export_text(myTree))

#prediction

predictions = myTree.predict(house.data)

print(type(predictions))
print(predictions)

#Evaluation

g = metrics.accuracy_score(house.target, predictions)
print(g) #1.0
print(metrics.f1_score(house.target, predictions, pos_label="True")) #1.0
print(metrics.precision_score(house.target, predictions, pos_label="False") #1.0
)

#Classess and probabilities
print("CLASSES: \n")
print(myTree.classes_)
pp = myTree.predict_proba(house.data)
print(pp)

# So far I have been practicing the function on the same data set I trained my tree with thus getting accuracy 1.0
# There are two common ways to evaluate a model's performance on unseen data: train-test split and cross-validation.
# with train-test split we split the data set into two parts then train with one and test with the other.
# A better method would be cross-validation
# cross-validation creates multiple train-test splits and averages the results to give a more robust estimate of model performance.
# In cross-validation, the dataset is split into 10 subsets (folds).
# The model is trained on 9 of them and tested on the remaining fold,
# and this process is repeated 10 times, using each subset as the test set once.
# The average of all these test scores provides a better estimate of the model's true performance.

print("CROSS-VALIDATION: \n")

dtc = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3)
y_scores = model_selection.cross_val_predict(dtc, house.data, house.target, method="predict_proba", cv=10)

print(y_scores.shape)
print(y_scores)
print(type(y_scores))

#ROC AUC
print("ROC AUC SCORE: \n")
ras = roc_auc_score(house.target, y_scores[:,1])
print(ras)
print("Plotting: \n")
fpr, tpr, th = roc_curve(house.target, y_scores[:,1],pos_label="True")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree Classifier (California Housing Dataset)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot(fpr,tpr,label="Decision Tree")
plt.legend()
plt.show()