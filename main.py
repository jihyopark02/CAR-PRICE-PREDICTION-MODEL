import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("Clean Data_pakwheels.csv")

# df.dropna(axis=0, inplace=True)

X = df.drop(['Price', 'Unnamed: 0'], axis=1)
y = df['Price']

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=0)

num_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
cat_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]

num_transformer = StandardScaler()
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)

model = Pipeline([
    ('processor', preprocessor),
    ('model', XGBRegressor(n_estimators=500, learning_rate=0.2))
])

model.fit(X_train, y_train)

prediction_valid = model.predict(X_valid)
print("y_valid vs prediction")
print("MAE: ", mean_absolute_error(prediction_valid, y_valid))

prediction_valid_score = model.score(X_valid, y_valid)
print("Accuracy: ", prediction_valid_score)

fig, ax = plt.subplots(figsize=(9, 6))
y_valid_arr = np.array(y_valid)
plt.plot(y_valid_arr[-50:], marker='o', label='y_valid')
plt.plot(prediction_valid[-50:], marker='o', label='prediction')
plt.title('y_valid vs prediction')
plt.legend()
plt.show()

prediction_test = model.predict(X_test)
print("y_test vs prediction")
print("MAE: ", mean_absolute_error(prediction_test, y_test))


prediction_test_score = model.score(X_test, y_test)
print("Accuracy: ", prediction_test_score)

fig, ax = plt.subplots(figsize=(9, 6))
y_test_arr = np.array(y_test)
plt.plot(y_test_arr[-50:], marker='o', label='y_test')
plt.plot(prediction_test[-50:], marker='o', label='prediction')
plt.title('y_test vs prediction')
plt.legend()
plt.show()