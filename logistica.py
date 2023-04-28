import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#1.1
# Carga de datos
data = np.genfromtxt('framingham.csv', delimiter=',', skip_header=1)
data = data[~np.isnan(data).any(axis=1)]

#1.2
#ajustar modelo de regresion logistica
# Variable independiente - Nivel de colesterol total
X = data[:, 9].reshape(-1, 1)
# Variable dependiente - Riesgo de enfermedad coronaria en 10 anios
y = data[:, 15]

# Normalizacion de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Agregar columna de unos
X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))

# Creacion de modelo y ajuste a los datos
clf = LogisticRegression()
clf.fit(X_scaled, y)

# Prediccion y evaluacion del modelo
y_pred = clf.predict(X_scaled)
acc = accuracy_score(y, y_pred)

#1.3 Descenso del gradiente
def sigmoid(z):
    """Función sigmoide."""
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, num_iter, lr):
    """Regresión logística con descenso de gradiente."""
    # Agregamos una columna de unos para theta_0
    X = np.column_stack((np.ones(X.shape[0]), X))

    # Inicializamos los parámetros theta a cero
    theta = np.zeros(X.shape[1])

    # Descenso de gradiente
    for i in range(num_iter):
        # Calculamos la función sigmoide para todos los valores de X*theta
        h = sigmoid(np.dot(X, theta))

        # Calculamos el gradiente de la función de costo con respecto a theta
        gradient = np.dot(X.T, (h - y)) / y.size

        # Actualizamos los parámetros theta
        theta -= lr * gradient

    return theta

#1.4 
def predict(X, theta):
    """Predice las etiquetas para las características X utilizando los parámetros theta."""
    # Agregamos una columna de unos para theta_0
    X = np.column_stack((np.ones(X.shape[0]), X))

    # Calculamos la función sigmoide para todos los valores de X*theta
    h = sigmoid(np.dot(X, theta))

    # Etiquetas predichas
    y_pred = np.round(h)

    return y_pred


def cross_validation(X, y, k, num_iter, lr):
    """Realiza k-fold cross-validation para determinar el grado del polinomio."""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    best_accuracy = 0
    best_degree = 0
    for degree in range(1, 6):
        # Generamos las características polinomiales de grado degree
        X_poly = np.column_stack([X**d for d in range(1, degree+1)])

        # Realizamos k-fold cross-validation
        accuracies = []
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X_poly[train_idx], y[train_idx]
            X_val, y_val = X_poly[val_idx], y[val_idx]

            # Entrenamos el modelo de regresión logística
            theta = logistic_regression(X_train, y_train, num_iter, lr)

            # Evaluamos la precisión del modelo en el conjunto de validación
            y_pred = predict(X_val, theta)
            accuracy = np.mean(y_pred == y_val)
            accuracies.append(accuracy)

        # Calculamos la precisión promedio sobre todos los pliegues
        accuracy = np.mean(accuracies)

        # Actualizamos el mejor grado y precisión si encontramos una precisión mejor
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_degree = degree

    return best_degree, best_accuracy

#1.5
#Analis de hallazgos
# Gráfica de dispersión de los datos originales y la línea de regresión ajustada
plt.scatter(X[:,0], y)
plt.plot(X[:,0], clf.predict_proba(X_scaled)[:,1], color='red')
plt.xlabel('Nivel de colesterol total')
plt.ylabel('Riesgo de enfermedad coronaria en 10 años')
plt.show()

# Gráfica de la función sigmoide utilizada en el modelo
z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z), color='red')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.show()
