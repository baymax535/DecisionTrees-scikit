from sklearn.datasets import fetch_openml

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

from sklearn.tree import DecisionTreeClassifier





