import numpy as np
import matplotlib.pyplot as plt

with open('potential.in') as data:
    for line in data:
        print(line)

# Define constants. Add loop to read file for variables later.
K_B       = 10 #Boltzmann constant?
X_FB_min = 0.5
X_FB_max = 5.5
Y_FB_min = 0.5
Y_FB_max = 5.5

# Define flat bottom parameters for x and y.
# FB(x)
def FB_x(x):
    if x < X_FB_min:
        return (x-X_FB_min)**2
    elif x > X_FB_max:
        return (x-X_FB_max)**2
    else:
        return 0.0
# FB(y)
def FB_y(y):
    if y < Y_FB_min:
        return (y-Y_FB_min)**2
    elif y > Y_FB_max:
        return (y-Y_FB_max)**2
    else:
        return 0.0

# sum A_i * ...
def summation(x,y,A,X,Y,sigma_X,sigma_Y):
    # check that A,X,Y,sigma_X,sigma_Y have the same shape
    assert A.shape == X.shape == Y.shape == sigma_X.shape == sigma_Y.shape
    # exponent
    exponent = - np.square(x-X)/(2*np.square(sigma_X)) - np.square(y-Y)/(2*np.square(sigma_Y))
    # A dot exp(...) -> out is a scalar
    out      = np.dot(A,np.exp(exponent))
    return out

# internal energy function: pass parameters and receive a new function that will have parameters fixed
def U(A,X,Y,sigma_X,sigma_Y):
    def callable(x,y):
        return K_B * FB_x(x) + K_B * FB_y(y) + summation(x, y, A, X, Y, sigma_X, sigma_Y)
    return callable

# hardcoded for now but implement read file later for efficiency
A       = np.array([-0.5,1.0,-0.5,1.0,-1.0,1.0,-0.25,1.0,-0.5])
X       = np.array([1.0,3.0,4.75,1.25,3.0,4.75,1.25,4.0,4.33])
Y       = np.array([1.0,1.25,1.25,3.0,3.0,3.0,4.75,4.75,4.33])
sigma_X = np.array([0.1,0.5,0.1,0.5,0.1,0.5,0.5,0.5,0.1])
sigma_Y = np.array([0.1,0.5,0.1,0.5,0.1,0.5,0.5,0.5,0.1])

# Define intervals
X_range = np.arange(0.0,6.0,0.05) # for x
Y_range = np.arange(0.0,6.0,0.05) # for y
Z       = np.zeros((X_range.size,Y_range.size)) # z to be populated


# create a function from a function. U_test will only depend on x and y
U_test = U(A, X, Y, sigma_X, sigma_Y)

# loop over x and y
for i,x in enumerate(X_range):
    for j,y in enumerate(Y_range):
        Z[i,j] = U_test(x, y)

# plot equation
# create a meshgrid
X, Y = np.meshgrid(X_range, Y_range)
fig  = plt.figure(figsize=(10,10))
ax   = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, Y, Z)
ax.view_init(50,30) #for rotation
plt.show()