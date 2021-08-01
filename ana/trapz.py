#!/usr/bin/env python 
"""
https://en.wikipedia.org/wiki/Trapezoidal_rule

Integral ( f(x) ) dx   ~    (b - a) 0.5(f(a) + f(b)) 
  a->b 


https://numpy.org/doc/stable/reference/generated/numpy.trapz.html

"""
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

     x = np.arange(10, dtype=np.float64)
     y = x*x 
     i_ = lambda _:np.power(_, 3)/3. 

     area_approx = np.trapz( y, x )
     area = i_( x[-1] ) - i_(x[0]) 

     fig, ax = plt.subplots()
     fig.suptitle( " np.trapz(y,x)  area_approx : %10.4f  area : %10.4f  " % (area_approx, area) ) 
     ax.plot( x, y )
     ax.scatter( x, y )

     fig.show()



