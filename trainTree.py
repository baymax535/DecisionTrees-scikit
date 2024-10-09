from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics


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
