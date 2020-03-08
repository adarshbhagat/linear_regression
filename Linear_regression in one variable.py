from matplotlib import pyplot as plt
import numpy as np
no_of_data=int(input("enter the no. of data"))
print("enter the value of x that is to be displayed in x axis(dependent variable)")
x=list(map(int,input().split()))
print("enter the value of y that is to be displayed in y axis(independent variable)")
y=list(map(int,input().split()))
x_mean=float(sum(x)/no_of_data)
y_mean=float(sum(y)/no_of_data)
new_x=[]
new_y=[]
for i in range(len(x)):
    new_x.append(float(x[i]-x_mean))
for i in range(no_of_data):
    new_y.append(float(y[i]-y_mean))
for i in range(no_of_data):
    new_y[i]=new_y[i]*new_x[i]
for i in range(no_of_data):
    new_x[i]=new_x[i]*new_x[i]
slope_b1=float(sum(new_y)/sum(new_x))
constant_b0=y_mean-(slope_b1*x_mean)
print("equation :- "+"y = " + str(constant_b0) +" + "+str(slope_b1)+"x")
x_mean=min(x)-(no_of_data/20)
y_mean=max(x)+(no_of_data/20)
plt.scatter(x,y)
plt.plot([x_mean,y_mean],[constant_b0+(slope_b1*x_mean),constant_b0+(slope_b1*y_mean)])
plt.show()
