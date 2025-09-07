import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Part 1
# Function that normalizes features in training set to zero mean and unit variance.
def normalize_train(X_train):
    # Input
    # --------------------
    # X_train : Training data, a numpy array with shape (n_samples, n_features),
    #         where n_samples is the number of samples in the training data, and 
    #         n_features is the number of features in the data
    #
    # Output
    # --------------------
    # X : The normalized version of the feature matrix, a numpy array
    # trn_mean : The mean of each column in the training set, float
    # trn_std : The std dev of each column in the training set, float
    
    '''
    Fill in your code here
    '''
    trn_mean = np.mean(X_train, axis=0)
    trn_std = np.std(X_train, axis=0)
    trn_std[trn_std == 0] = 1  # Prevent division by zero
    X = (X_train - trn_mean) / trn_std
    return X, trn_mean, trn_std  
    

# Part 2
# Function that normalizes testing set according to mean and std of training set

def normalize_test(X_test, trn_mean, trn_std):
    
    # Input
    # --------------------
    # X_test : Testing data, a numpy array with shape (n_samples, n_features)
    # trn_mean : Mean of each column in training set, float
    # trn_std : Standard deviation of each column in training set, float
    #
    # Output
    # --------------------
    # X : The normalized version of the feature matrix, X_test

    '''
    Fill in your code here
    '''
    trn_std[trn_std == 0] = 1  # Prevent division by zero
    X = (X_test - trn_mean) / trn_std
    return X 

    


# Part 3
# Function to return a numpy array generated with `np.logspace` with a length
# of 51 starting from 1E^-1 and ending at 1E^3
def get_lambda_range():
    
    # Input
    # --------------------
    # Nothing
    #
    # Output
    # --------------------
    # lmbda : numpy array of logarithmically spaced values
    
    '''
    Fill in your code here
    '''
    return np.logspace(-1, 3, 51)


# Part 4
# Function that trains a ridge regression model on the input dataset with lambda=1
def train_model(X_train, y_train, l):
    # Input
    # --------------------
    # X_train : Feature matrix
    # y_train : Target variable vector
    # l : Regularization parameter
    #
    # Output
    # --------------------
    # model : A numpy object containing the trained model

    '''
    Fill in your code here
    '''
    model = Ridge(alpha=l, fit_intercept=True)  # Ensure model fits intercept
    model.fit(X_train, y_train)
    return model


# Part 5
# Function that calculates the mean squared error of the model on the input dataset
def error(X, y, model):
    
    # Input
    # ------------------
    # X : Feature matrix
    # y : Target variable vector
    # model : Numpy model object
    #
    # Output
    # ------------------
    # mse : Mean squared error

    '''
    Fill in your code here
    '''
    y_pred = model.predict(X)
    return np.mean((y - y_pred) ** 2)

def main():
    # Importing dataset
    # step 1 : read csv
    df = pd.read_csv("AAPL.csv")
    # step 2 : identify the column(s) we want to remove
    remove_features = ["Date"]
    # step 3: create extra column for prediction by shifting
    # rows of `Close` columns by one to obtain next day's closing price
    df["Prediction"] = df["Close"].shift(-1)
    # step 4: drop the last row because it would have invalid value after the shift.
    df.dropna(inplace=True)
    # step 5: remove the columns identified in step 2
    df.drop(columns=remove_features, inplace=True)
    # step 6: create X by dropping the `Prediction` column
    X = df.drop(columns=["Prediction"]).values
    # step 7: Store `Prediction` column in y array
    y = df["Prediction"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    # Problem 1:
    # Complete the function 'normalize_train' above
   
    # Problem 2:
    # Complete the function 'normalize_test' above


    # Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)


    # Problem 3: 
    # Complete the function 'get_lambda_range'.
    
    # Problem 4:
    # Complete the function 'train_model' above.

    # Problem 5:
    # Complete the function 'error' above.

    # Define the range of lambda to test
    lmbda = get_lambda_range()
    # uncomment the below line to help verify your answer is correct.
    # lmbda = [1, 500, 1000, 2500]
    # make sure to re-comment above line so that the lambda checked during grading is from 'get_lambda_range'
    MODEL = []
    MSE = []
    for l in lmbda:
        # Train the regression model using a regularization parameter of l
        model = train_model(X_train, y_train, l)

        # Evaluate the MSE on the test set
        mse = error(X_test, y_test, model)

        # Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    # Part 6
    # Plot the MSE as a function of lmbda
    # Note that the code to find MSE has already been completed in part 5.
    ''' 
    fill in your code below 
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(lmbda, MSE, linestyle='-', color='black')
    plt.xscale("log")
    plt.xlabel("Regularization Parameter Lambda")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE vs. Lambda")
    plt.show()

    # Find best value of lmbda in terms of MSE
    # Record this value on the writeup
    ''' 
    fill in your code below 
    '''
    ind = np.argmin(MSE)
    lmda_best, MSE_best, model_best = lmbda[ind], MSE[ind], MODEL[ind]

    # Part 7
    # Using the best model found above, write out the model coefficients and intercept
    # Record this value on the writeup
    ''' 
    fill in your code below 
    '''
    print(f"Best lambda tested is {lmda_best}, which yields an MSE of {MSE_best}")
    print(f"Best Model Coefficients: {model_best.coef_}")
    print(f"Best Model Intercept: {model_best.intercept_}")

    # Part 8
    # Load GOOG.csv 
    # This process will be similar to steps 1-7 where AAPL.csv is loaded
    # Note that you should NOT call 'train_test_split' in this part.
    ''' 
    fill in your code below 
    '''
    df_goog = pd.read_csv("GOOG.csv")
    df_goog["Prediction"] = df_goog["Close"].shift(-1)
    df_goog.dropna(inplace=True)
    df_goog.drop(columns=["Date"], errors="ignore", inplace=True)
    X_goog = normalize_test(df_goog.drop(columns=["Prediction"]).values, trn_mean, trn_std)

    # normalize X similar to X_test
    y_goog = df_goog["Prediction"].values
    y_pred = model_best.predict(X_goog)

    # plot y and y_hat
    plt.figure(figsize=(8, 6))
    plt.plot(y_goog, label="actual", color="blue")
    plt.plot(y_pred, label="prediction", color="orange")
    plt.legend()
    plt.show()
    return model_best


if __name__ == "__main__":
    model_best = main()
