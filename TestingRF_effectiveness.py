import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np


"""Remember that this is a basic implementation. You might want to consider the following to improve the model:
1) Feature scaling or normalization if the features are on different scales.
2) Hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV.
3) Handling any missing values or outliers in the dataset.
4) Cross-validation for more robust performance estimation."""

def TrainRFR(train, test):# Load the CSV file
    
    MinMaxscaler = MinMaxScaler() #use the same scaling instance the whole time
    #to pull out the features (X) and target variable (Y = band_gap)
    def GetXandY(df):
        X = df.iloc[:, 1:-1]  #Exclude the first column (as it's the material identifier) and the target variable
        Y = df['band_gap']
        if trainORtest.lower() == 'train': #if training data then fit the scaling and apply it
            X = MinMaxscaler.fit_transform(X)
        elif trainORtest.lower() == 'train': #if test data, then apply the scaling that was already established 
            X = MinMaxscaler.transform(X)
        else: #Sometime I might not want to do scaling
            pass  
        return X, Y

    #Create and train the RandomForestRegressor:
    x_train, y_train = GetXandY(pd.read_csv(train), trainORtest='train')
    rf_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=4, min_samples_split=8, max_features='sqrt', criterion='squared_error')
    rf_model.fit(x_train, y_train)

    #Make predictions and evaluate the model:
    x_test, y_test = GetXandY(pd.read_csv(test), trainORtest='test')
    y_pred = rf_model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    pA = ((y_pred-y_test)/y_test)*100

    print(f"Percent Accuracy: {pA}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared Score: {r2}")

    #check feature importances:
    feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print(feature_importance)

if __name__ == "__main__":
    ToData = '/Users/chimamcgruder/Work_General/ClimateBase/Materials/MLtests/mp_139K_11feat/DataSubset_3000training_10crossVal/'
    TrainRFR(ToData+'trainingDat_sub0.csv', ToData+'testDat.csv')