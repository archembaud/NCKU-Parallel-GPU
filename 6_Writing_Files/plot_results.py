import matplotlib.pyplot as plt
from numpy import genfromtxt

# Load results
result = genfromtxt('results.csv', delimiter='\t')

# Save into familiar variables
a = result[:, 0]
b = result[:, 1]
c = result[:, 2]

plt.plot(a, c)  
plt.xlabel('a')  
plt.ylabel('c')  
plt.title('Demo 6 Graph - a vs c') 

# Save the figure to file
plt.savefig("results.jpg")