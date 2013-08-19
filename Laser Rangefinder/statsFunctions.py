import scipy
import numpy

def fitInverseFunction(x, y):
    recipricate(y)
    (ar,br) = scipy.polyfit(y, x, 1)
    
    xr=scipy.polyval([ar,br],y)
    R = calculateR(x,y)
    
    return (ar,br,R**2)

def recipricate(arr):
    for i in range(len(arr)):
        arr[i] = 1/arr[i]

def calculateR(x, y):
    n = len(x)
    return (n*sum(x*y)-sum(x)*sum(y))/numpy.sqrt((n*sum(x*x)-sum(x)*sum(x))*(n*sum(y*y)-sum(y)*sum(y)))
