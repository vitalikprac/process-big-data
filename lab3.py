# Лаба3 (навчання без вчителя).
# Обрати датасет в якому наявні вхідні дані. 
# Зробити модель машинного навчання, яка буде передбачати/визначати дані по вхідним даним. 
# Оцінити точність моделі по валідаційній вибірці.
# (приклад датасету - Iris).

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

iris_data = load_iris()
x = iris_data["data"]
y = iris_data["target"]

x_train, x_validation, y_train, y_validation = train_test_split(x,y, test_size=0.3, random_state=50)
kmeans = KMeans(n_init=10, n_clusters=3)
kmeans.fit_predict(x_train)

accuracy_score = kmeans.score(x_validation)
print(f"Validation accuracy (opposite of the value of X on the K-means objective): {accuracy_score:.4f}")
