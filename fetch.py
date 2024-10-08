from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


print("\nCREDIT: \n")
credit = fetch_openml(data_id=31)
print(credit.details, "\n")
credit.frame.info()
print(credit.data)
print(credit)
print(credit.target)

print("\nMUSHROOM: \n")
mushroom = fetch_openml(data_id=24)
print(mushroom.details, "\n")
mushroom.frame.info()
print(mushroom.data)
print(mushroom)
print(mushroom.target)

myTree = DecisionTreeClassifier(criterion="entropy") #untrained decision tree with entropy criteria. default uses Gini index

myTree.fit(credit.data, credit.target) #trained

print(tree.export_text(myTree))

#Update: Line 24 did not train because I am using Datasets with Nominal Features.


