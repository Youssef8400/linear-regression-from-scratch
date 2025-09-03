import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data=pd.read_csv("BostonHousing.csv")


print(data.columns)
print(data.isnull().sum())
print(data.shape)
print(data.corr())

X = data.loc[:, ["rm" , "ptratio" , "lstat"]]
Y = data.loc[:, "medv"]


def train_test_split_manual(x , y=None , test_size = 0.3 , random_state = None  , shullfe=None):
    n = len(X)
    if n == 0 :
        print("X est vide")

    if isinstance(test_size , float):
        n_test = int(round(n*test_size))
    else : 
        n_test = int(test_size)

    if not (0 < n_test < n):
        print("test_size doit être entre 1 et n-1 (ou 0<frac<1).")

    rng = np.random.RandomState(random_state)
    indices = np.arange(n)

    if shullfe:
        rng.shuffle(indices)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    def _index(obj, idx):
        if hasattr(obj, "iloc"):
            return obj.iloc[idx]
        return np.asarray(obj)[idx]
    
    x_train = _index(X , train_idx)
    x_test = _index(X , test_idx)


    if Y is None:
        return x_train, x_test

    y_train = _index(Y , train_idx)
    y_test = _index(Y , test_idx)       


    return x_train , x_test , y_train , y_test


class YsfLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self,X ,Y):
        X = np.asarray(X)
        Y = np.asarray(Y)

        X_b = np.c_[np.ones((X.shape[0] , 1)) , X]


        beta = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ Y)
        
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

    def predict(self , X):
        X = np.asarray(X)
        X_b = np.c_[np.ones((X.shape[0] , 1)),X]
        return X_b @ np.r_[self.intercept_ , self.coef_]
    

x_train , x_test , y_train , y_test = train_test_split_manual(X , Y , test_size=0.3 ,random_state=0)


model = YsfLinearRegression()
model.fit(x_train , y_train)

y_pred = model.predict(x_test)

mse = np.mean((y_test - y_pred)**2)
rmse = np.sqrt(mse)

print("Intercept :", model.intercept_)
print("Coefficients :", model.coef_)
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")

plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Réel")
plt.ylabel("Prédiction")
plt.title("Prédictions vs Réalité")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
plt.show()

