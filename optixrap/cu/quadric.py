#/usr/bin/env python
"""
Investigate Homogenous matrix approach to quadric intersection,
following:

* Ken Chan, A Simple Mathematical Approach for Determining Intersection of Quadratic Surfaces
* ~/opticks_refs/Chan_4D_quad_intersect_only.pdf

Hyperboloid :  x**2 + y**2 - z**2 -  h**2 = 0 
Sphere      :  x**2 + y**2 + z**2  - r**2  = 0 


* [x,y,z,1].T  M  [x,y,z,1] = 0 


In [22]: sp
Out[22]: 
Matrix([
[1, 0, 0,     0],
[0, 1, 0,     0],
[0, 0, 1,     0],
[0, 0, 0, -r**2]])

In [23]: hy = Matrix([[1,0,0,0],[0,1,0,0,],[0,0,-1,0],[0,0,0,-h*h]])

In [24]: hy
Out[24]: 
Matrix([
[1, 0,  0,     0],
[0, 1,  0,     0],
[0, 0, -1,     0],
[0, 0,  0, -h**2]])

In [25]: hy.inv()
Out[25]: 
Matrix([
[1, 0,  0,       0],
[0, 1,  0,       0],
[0, 0, -1,       0],
[0, 0,  0, -1/h**2]])

In [26]: sp.inv()
Out[26]: 
Matrix([
[1, 0, 0,       0],
[0, 1, 0,       0],
[0, 0, 1,       0],
[0, 0, 0, -1/r**2]])

In [27]: sp.det()
Out[27]: -r**2

In [28]: hy.det()
Out[28]: h**2



"""


from sympy import symbols, eye, Matrix

h,r = symbols("h,r")


# from [x,y,z,1] 
sp = Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-r*r]])
hy = Matrix([[1,0,0,0],[0,1,0,0,],[0,0,-1,0],[0,0,0,-h*h]])




