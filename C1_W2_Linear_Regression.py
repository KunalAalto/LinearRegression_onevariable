



# Linear regresson with one variable. It can be utilised for developing linear regression models with one variable.

# Importing the important libraries

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# Loading the data from the file. You can use any files.
x_train, y_train = load_data()



# print x_train 
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5]) 

# print y_train
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])  

print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))


# Create a scatter plot of the data. To change the markers to red "x",
# I used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r') 
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()


# Making a cost function to compute cost for linear regression. Here, w and b are parameters for the model

def compute_cost(x, y, w, b): 

    # number of training examples
    m = x.shape[0] 
    
    total_cost = 0
    cost_sum= 0
    
    for i in range (m):
        f_wb= w*x[i] + b
        cost_i = (f_wb - y[i])**2
        cost_sum = cost_sum + cost_i
    total_cost = (1/(2*m)) *(cost_sum)
    return total_cost # returns the total cost.




# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

# Making a function to calculate gradient for linear regression. dj_dw is the gradient of the cost w.r.t. the parameters w. dj_db is the gradient of the cost w.r.t. the parameter b.
  
def compute_gradient(x, y, w, b): 
    
    
    
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
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


# Calculating gradient descent. alpha is learning rate.
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 

    
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
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
        if i<100000:      # prevent resource exhaustion 
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

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b


#

# Plot the linear fit
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')