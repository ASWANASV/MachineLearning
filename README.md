# MachineLearning
1. Loading and Preprocessing (5 marks)
 Load the dataset and perform necessary preprocessing steps.
ANS:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


print(df.isnull().sum())


#Handle missing values
df_numeric = df.select_dtypes(include=['int64', 'float64'])
df_numeric = df_numeric.fillna(df_numeric.mean())


df_non_numeric = df.select_dtypes(include=['object'])
df_non_numeric = df_non_numeric.fillna('Unknown')


df = pd.concat([df_numeric, df_non_numeric], axis=1)


print(df.head())








#Encode categorical variables
df['fuel_type'] = pd.get_dummies(df['fuel_type'], drop_first=True)


#Scale numerical features
scaler = StandardScaler()
df[['engine_size', 'horsepower', 'price']] = scaler.fit_transform(df[['engine_size', 'horsepower',  'price']])


#Split the data into training and testing sets
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
2. Model Implementation (10 marks)
 Implement the following five regression algorithms:
ANS:
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

1) Linear Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

string_cols = df.select_dtypes(include=['object']).columns
print("columns with string values:")
print(string_cols)

for col in string_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

2) Decision Tree Regressor

dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)


3) Random Forest Regressor

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)


4) Gradient Boosting Regressor

gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

5) Support Vector Regressor

svr_model = SVR()
svr_model.fit(X_train, y_train)


3. Model Evaluation (5 marks)
Compare the performance of all the models based on R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).
Identify the best performing model and justify why it is the best.
ANS:
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


#Linear Regression
y_pred_lr = lr_model.predict(X_test)
print("Linear Regression:")
print("R-squared:", r2_score(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))


#Decision Tree Regressor
y_pred_dt = dt_model.predict(X_test)
print("\nDecision Tree Regressor:")
print("R-squared:", r2_score(y_test, y_pred_dt))
print("MSE:", mean_squared_error(y_test, y_pred_dt))
print("MAE:", mean_absolute_error(y_test, y_pred_dt))


#Random Forest Regressor
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Regressor:")
print("R-squared:", r2_score(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))


#Gradient Boosting Regressor
y_pred_gb = gb_model.predict(X_test)
print("\nGradient Boosting Regressor:")
print("R-squared:", r2_score(y_test, y_pred_gb))
print("MSE:", mean_squared_error(y_test, y_pred_gb))
print("MAE:", mean_absolute_error(y_test, y_pred_gb))


#Support Vector Regressor
y_pred_svr = svr_model.predict(X_test)
print("\nSupport Vector Regressor:")
print("R-squared:", r2_score(y_test, y_pred_svr))
print("MSE:", mean_squared_error(y_test, y_pred_svr))
print("MAE:", mean_absolute_error(y_test, y_pred_svr))




*To determine which model is the best
   1. *R-squared (R²)*: Higher values are better. A value close to 1 indicates that the       model is a good fit for the data.
    2. *Mean Squared Error (MSE)*: Lower values are better. A value close to 0 indicates that the model is making accurate predictions.
    3. *Mean Absolute Error (MAE)*: Lower values are better. A value close to 0 indicates that the model is making accurate predictions.
Since 
#Random Forest Regressor is the best



4. Feature Importance Analysis (2 marks)
Identify the significant variables affecting car prices (feature selection)
ANS:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


#Feature importance analysis
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
feature_importances = rf_model.feature_importances_
feature_importances_df = pd.DataFrame({'feature': X_train.columns, 'importance': feature_importances})
feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
print(feature_importances_df)


plt.figure(figsize=(10, 6))
plt.bar(feature_importances_df['feature'], feature_importances_df['importance'])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()


5. Hyperparameter Tuning (2 marks):
Perform hyperparameter tuning and check whether the performance of the model has increased.
ANS:
from sklearn.model_selection import GridSearchCV


Define the hyperparameter tuning space
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}


Initialize the Random Forest Regressor
rf_model = RandomForestRegressor()


Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)


Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)


Print the best parameters and the corresponding best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


Print the complete grid search results
cv_results = grid_search.cv_results_
for i in range(len(cv_results['params'])):
    print(f"Parameters: {cv_results['params'][i]}")
    print(f"Mean Test Score: {cv_results['mean_test_score'][i]}")
    print(f"Rank: {cv_results['rank_test_score'][i]}")
    print()


