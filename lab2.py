# Lab 2
# Лаба 1 тільки використовуючи дерева 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris_data = load_iris()
x = iris_data["data"]
y = iris_data["target"]

x_train, x_remaining, y_train, y_remaining = train_test_split(x,y, test_size=0.3, random_state=50)
x_validation, x_test, y_validation, y_test = train_test_split(x_remaining, y_remaining, test_size=1/3, random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

val_predictions = model.predict(x_validation)
accuracy = accuracy_score(y_validation, val_predictions)
print(f"Validation accuracy: {accuracy:.4f}")

print("Test values:")
test_predictions = model.predict(x_test)
for idx, x  in enumerate(test_predictions):
    print(f"Actual value: {y_test[idx]} and predicted: {x}")