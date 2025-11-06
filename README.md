# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1.Import the required libraries.

2.Load the Iris dataset using load_iris() from sklearn.

3.Create a DataFrame and display the first few rows.

4.Split the dataset into training and testing sets.

5.Standardize the data using StandardScaler().

6.Create and train the model using SGDClassifier().

7.Predict the output for the test data.

8.Calculate and print the accuracy and classification report.

9.Plot the confusion matrix using a heatmap.

10.Plot a scatter graph to visualize the species.

11.Test the model with a new sample input.

## Program:
```

Program to implement the prediction of iris species using SGD Classifier.
Developed by: KOMALAVARSHINI.S
RegisterNumber: 212224230133

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


iris = load_iris()


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("📘 Iris Dataset (first 5 rows):")
print(df.head())


X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train)


y_pred = sgd_clf.predict(X_test)


print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))


cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix for Iris SGD Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


plt.figure(figsize=(8, 6))
for i, species in enumerate(np.unique(y)):
    plt.scatter(
        df[df["species"] == i]["petal length (cm)"],
        df[df["species"] == i]["petal width (cm)"],
        label=iris.target_names[i]
    )

plt.title("Iris Flower Classes (Petal Length vs Petal Width)")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.grid(True)
plt.show()


sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
sample_scaled = scaler.transform(sample)
predicted_class = sgd_clf.predict(sample_scaled)
```
## Output:
### Iris Dataset (first 5 rows):
<img width="736" height="125" alt="image" src="https://github.com/user-attachments/assets/c5fc8739-c210-41be-b611-e36a4bdb8ff2" />

### species species_name: 
<img width="663" height="108" alt="image" src="https://github.com/user-attachments/assets/2b7fa5f6-0912-4e15-a396-92e1239de780" />

### Model Accuracy:
<img width="600" height="32" alt="image" src="https://github.com/user-attachments/assets/6426a632-3a40-4f57-bf8d-4b6818810be9" />

### Classification Report:
<img width="598" height="184" alt="image" src="https://github.com/user-attachments/assets/6ac2d1c2-940f-49d0-a4d1-ef74741cff4a" />

### Confusion Matrix for Iris SGD Classifier:
<img width="656" height="370" alt="image" src="https://github.com/user-attachments/assets/f332ed63-05c4-421f-9303-a065fee9ad74" />

### Iris Flower Classes (Petal Length vs Petal Width):
<img width="759" height="520" alt="image" src="https://github.com/user-attachments/assets/89cc5e65-5195-4134-a963-c3547dcf82d6" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
