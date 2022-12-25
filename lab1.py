# Lab 1
# Обрати датасет в якому наявні вхідні дані та вихідні (очікуваний результат).
# Датасет поділити на 3 частини (70%:20%:10%, навчальний, валідаційний, тестувальний набір).
# Зробити модель машинного навчання (навчання з вчителем, крім дерев), яка буде передбачати/визначати дані по вхідним даним. 
# Оцінити точність моделі по валідаційній вибірці.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

iris_data = load_iris()
x = iris_data["data"]
y = iris_data["target"]

x_train, x_remaining, y_train, y_remaining = train_test_split(x,y, test_size=0.3, random_state=50)
x_validation, x_test, y_validation, y_test = train_test_split(x_remaining, y_remaining, test_size=1/3, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

val_predictions = model.predict(x_validation)
val_error = mean_squared_error(y_validation, val_predictions)
print(f"Validation Mean squared error: {val_error:.4f}")

print("Test values:")
test_predictions = model.predict(x_test)
for idx, x  in enumerate(test_predictions):
    print(f"Actual value: {y_test[idx]} and predicted: {x}")