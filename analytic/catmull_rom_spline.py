#!/usr/bin/env python
"""
catmull_rom_spline.py
======================

::

   epsilon:opticks blyth$ opticks-rl sympy 


* https://www.cs.cmu.edu/~fp/courses/graphics/asst5/catmullRom.pdf

* https://qroph.github.io/2018/07/30/smooth-paths-using-catmull-rom-splines.html

* https://www.cs.cmu.edu/~fp/courses/graphics/asst5/catmullRom.pdf

* https://dev.to/ndesmic/splines-from-scratch-catmull-rom-3m66


"""

import numpy as np, sympy as sp, matplotlib as mp
from sympy.utilities.lambdify import lambdify  
SIZE = np.array([1280, 720])

def pp(q, q_label="?", note=""):
    if type(q) is np.ndarray:
        q_type = "np"
    elif q.__class__.__name__  == 'MutableDenseMatrix':
        q_type = "sp.MutableDenseMatrix"
    else:
        q_type = "?"
    pass
    print("\n%s : %s : %s : %s \n" % (q_label, q_type, str(q.shape), note) )

    if q_type.startswith("sp"):
        sp.pprint(q)
    elif q_type.startswith("np"):
        print(q)
    else:
        print(q)
    pass


if __name__ == '__main__':

    """
  
      0,1        1,1
        +--------+
        |        |   
        |        |   
        |        |   
        +--------+
      0,0       1,0

    """


    points = np.zeros( (16, 3) )
    for j in range(len(points)):
        angle = 2*np.pi*float(j)/(len(points))
        # when looped avoiding the repeated point from 0,2pi degeneracy 
        # avoids the joint being visible
    
        points[j] = [np.cos(angle), np.sin(angle), 0]
    pass

    #points = np.array([ [0,0,0], [1,0,0], [0,1,0], [1,1,0] ])  


    A = 0.5   # eg 0:straight lines, 0.5 default,smoothest? 1:wiggly  "tension"

    assert len(points) >= 4 
    numseg = len(points)-3    # sliding window of 4 points for each segment  

    looped = True
    if looped: numseg += 3    

    segnum = 100              # number of points to interpolate each segment into 
    tt = np.linspace(0,1, segnum)

    xpp = np.zeros( (numseg*segnum,3) )  


    u, a  = sp.symbols("u a")
    M = sp.Matrix([
           [  0   , 1    , 0      ,  0 ],
           [  -a  , 0    , a      ,  0 ],
           [  2*a , a - 3, 3 - 2*a, -a ],  
           [  -a  , 2 - a, a - 2  ,  a ]
           ])

    pp(M,"M")

    U = sp.Matrix( [1, u, u*u, u*u*u ] )     # column vector
    V = sp.Matrix( [[1, u, u*u, u*u*u ]] )   # row vector
    pp(U,"U")
    pp(V,"V")
    pp(U.T,"U.T")     
    pp(V.T,"V.T")   
    VM = V*M
    pp(VM,"VM")     
    pp(VM.T,"VM.T")     


    VMa = VM.subs([(a,A)])   
    numpoi = len(points) 

    for iseg in range(numseg):

        i0 = (iseg+0) % numpoi
        i1 = (iseg+3) % numpoi
        #p = points[i0:i1+1]   # python one beyond index
        p = np.take( points, np.arange(iseg,iseg+4), mode='wrap', axis=0 )

        ## hmm now to loop the array indices in numpy ?
        ## https://stackoverflow.com/questions/28398220/circular-numpy-array-indices

        print(" iseg %2d i0 %3d i1 %3d numpoi %3d p %s " % (iseg, i0,i1,numpoi, str(p.shape) ) )

        # ordinarily, when not looped the modulus will not kick in 
        #
        #     numseg : numpoi-3     (sliding window of 4 points for each segment)
        #   max iseg : numseg-1  = numpoi-4
        #      

        VMa_p = VMa*p 
        #pp(VMa_p, "VMa_p")
        fn = lambdify(u, VMa_p,'numpy')  

        for i in range(len(tt)):
           xpp[iseg*segnum+i] = fn(tt[i])     
           # how to directly pull an array of arrays here ? avoiding the python loop 
        pass    
    pass



    fig, ax = mp.pyplot.subplots(figsize=SIZE/100.) 
    fig.suptitle("analytic/catmull_rom_spline.py")
    ax.set_aspect('equal')
    ax.plot( xpp[:,0], xpp[:,1], label="xpp"  )
    ax.scatter( points[:,0], points[:,1], label="points" )
    ax.legend()
    fig.show()

