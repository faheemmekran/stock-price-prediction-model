import numpy as np
import matplotlib.pyplot as plt 


# Function that returns fitted model parameters to the dataset at datapath for each choice in degrees.
# 

def main(datapath, degrees):
    
    # Input
    # --------------------
    # datapath : A string specifying a .txt file 
    # degrees : A list of positive integers.
    #    
    # Output
    # --------------------
    # paramFits : a list with the same length as degrees, where paramFits[i] is the list of
    #             coefficients when fitting a polynomial of d = degrees[i].
    
    paramFits = []

    file = open(datapath, 'r')
    data = file.readlines()
    x = []
    y = []
    for line in data:
        [i, j] = line.split()
        x.append(float(i))
        y.append(float(j))
        
    # iterate through each n in the list degrees, calling the feature_matrix and least_squares functions to solve
    # for the model parameters in each case. Append the result to paramFits each time.
    '''
    fill in your code here
    '''
    for d in degrees:
        X = feature_matrix(x, d)  
        B = least_squares(X, y)   
        paramFits.append(B)      
        

    return paramFits


# Function that returns the feature matrix for fitting a polynomial of degree d based on the explanatory variable
# samples in x.
#

def feature_matrix(x, d):
    
    # Input
    # --------------------
    # x: A list of the independent variable samples
    # d: An integer
    #
    # Output
    # --------------------
    # X : A list of features for each sample, where X[i][j] corresponds to the jth coefficient
    #     for the ith sample. Viewed as a matrix, X should have dimension (samples, d+1).

    # There are several ways to write this function. The most efficient would be a nested list comprehension
    # which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    # Please be aware of which matrix colum corresponds to which degree polynomial when completing the writeup.
    '''
    fill in your code here
    '''
    X = []
    for xi in x:
        row = [xi ** power for power in range(d, -1, -1)]
        X.append(row)
    return X


# Function that returns the least squares solution based on the feature matrix X and corresponding target variable samples in y.
def least_squares(X, y):
    # Input
    # --------------------
    # X : A list of features for each sample
    # y : a list of target variable samples.
    #
    # Outut
    # --------------------
    # B : a list of the fitted model parameters based on the least squares solution.
    
    X = np.array(X)
    y = np.array(y)

    # Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    '''
    fill in your code here
    '''
    X = np.array(X)
    y = np.array(y)

    # Solve the normal equation (X^T * X) B = X^T * y
    B = np.linalg.inv(X.T @ X) @ X.T @ y
    return B



if __name__ == "__main__":
    datapath = "poly.txt"

    file = open(datapath, 'r')
    data = file.readlines()
    x = []
    y = []
    for line in data:
        [i, j] = line.split()
        x.append(float(i))
        y.append(float(j))

    ### Problem 1 ###
    # TODO: Complete 'main, 'feature_matrix', and 'least_squares' functions above


    ### Problem 2 ###
    # The paramater values for degrees 2 and 4 have been provided as test cases in the README.
    # The output should match up to at least 3 decimal places rounded 

    
    # Write out the resulting estimated functions for each d.
    degrees = [1,2,3, 4,5,6] # TODO: Update the degrees,d to include 1, 3, 5 and 6. i.e. [1,2,3,4,5,6] and
    paramFits = main(datapath, degrees)
    for idx in range(len(degrees)):
        print("y_hat(x_"+str(degrees[idx])+")")
        print(paramFits[idx])
        print("****************")

    ### Problem 3 ###
    # TODO: Visualize the dataset and these fitted models on a single graph
    # Use the 'scatter' and 'plot' functions in the `matplotlib.pyplot` module.

    # Draw a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='black', label='data')
    x_range = np.linspace(min(x), max(x), 100)
    x.sort() # Ensures the scatter plot looks nice

    colors = ['orange', 'red', 'magenta', 'cyan', 'blue', 'green']
    for i, params in enumerate(paramFits):  # Use enumerate() to avoid index errors
        x_range = np.linspace(min(x), max(x), 100)  # Smooth range for plotting
        X_test = feature_matrix(x_range, degrees[i])  # Generate feature matrix
        y_pred = np.dot(X_test, params)
        plt.plot(x_range, y_pred, label=f"d = {degrees[i]}", color=colors[i], linewidth=2)

    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid()
    plt.show(block=True)
    plt.close()
    ### Problem 4 ###
    # TODO: when x = 4; what is the predicted output
    # Use the degree that best matches the data as determined in Problem 3 above.
    
    '''
    fill in your code here
    '''
    x_test = 4
    best_degree = 4  # Assume the best fit polynomial degree is 4 (adjust based on visualization)
    best_params = paramFits[degrees.index(best_degree)]
    
    X_test = feature_matrix([x_test], best_degree)
    y_hat = np.dot(X_test, best_params)
    
    print(f"Predicted value for x = {x_test}: {y_hat[0]:.3f}")

