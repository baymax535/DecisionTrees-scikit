from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

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
