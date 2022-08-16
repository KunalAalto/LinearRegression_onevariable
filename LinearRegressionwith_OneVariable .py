# Linear regression with one variable. This code was developed for the housing price prediction.

# importing important python libraries

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
%matplotlib inline

# load the dataset.
x_train, y_train = np.loadtxt('dataset.txt', unpack=True)

#Check the dimensions of data  and calculate total number of training examples.
print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))

# Making a scatter plot of the data
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Price vs. size of the house")
# Set the y-axis label
plt.ylabel('Price in $10,000')
# Set the x-axis label
plt.xlabel('Size of house in 1000')
plt.show()

# Making a function to compute the cost for linear regression.

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Arguments:
        x (ndarray): Shape (m,) Input to the model (size of houses) 
        y (ndarray): Shape (m,) Label (price of houses)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # initializing parameters
    total_cost = 0
    cost_sum= 0
    
    for i in range (m):
        f_wb= w*x[i] + b
        cost_i = (f_wb - y[i])**2
        cost_sum = cost_sum + cost_i
    total_cost = (1/(2*m)) *(cost_sum)
    
    return total_cost
    
# Making a function for computing gradient
def compute_gradient(x, y, w, b): 
    """
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    # initializing dj_db and dj_dw
    dj_dw = 0
    dj_db = 0
    
    for i in range (m) :
        f_wb = w*x[i] + b 
        dj_dw_i = (f_wb - y[i])*x[i]
        dj_db_i = f_wb -y[i]
        
        dj_dw = dj_dw + dj_dw_i
        dj_db = dj_db + dj_db_i 
        
    dj_dw = dj_dw / m 
    dj_db = dj_db / m       
   
        
    return dj_dw, dj_db



# Making a funciton to calculate gradient descent for linear regression
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha :  Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w :  Shape (1,) Updated values of parameters of the model after running gradient descent
      b : Updated value of parameter of the model after running gradient descent
    """
    
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration 
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<200000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing
    
    
# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.


iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)

print("w,b found by gradient descent:", w, b)


# Plot for linear fit
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b
# Plot the linear fit
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(x_train, y_train, marker='x', c='r') 

#title
plt.title("Price vs. size of the house")
# Set the y-axis label
plt.ylabel('Price in $10,000')
# Set the x-axis label
plt.xlabel('Size of house in 1000s')
plt.show()  


