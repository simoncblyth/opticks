#!/usr/bin/env python
"""
"""
import numpy as np


def make_normal( a, b, c, dtype=np.float32):
    """
              C
        C-A  / \
         /  /   \
           /     \
          /       \
         A_________B
              ->
             B-A 

         n = (B - A) x (C - A) 
 
      * normal outwards
      * order A, B, C as anti-clockwise as viewed from outside

    """
    a = np.asarray(a, dtype=dtype)
    b = np.asarray(b, dtype=dtype)
    c = np.asarray(c, dtype=dtype)

    assert a.shape == (3,)
    assert b.shape == (3,)
    assert c.shape == (3,)
    
    ba = b - a
    ca = c - a

    n = np.cross( ba, ca )
    nn = np.sqrt(np.dot(n,n))

    #print "a %40s b %40s c %40s " % (a,b,c)
    n /= nn
    #print "ba %40s ca %40s n %40s nn %s " % (ba, ca, n, nn )

    return n  
    


def make_plane( normal, point, dtype=np.float32 ):
    """
    Form plane from its normal and any point that lies on the plane   
    """
    normal = np.asarray(normal, dtype=dtype)
    point = np.asarray(point, dtype=dtype)

    assert normal.shape == (3,)
    assert point.shape == (3,)
    normal /= np.sqrt(np.dot(normal,normal))

    d = np.dot(normal, point)  # distance of point from origin 
    plane = np.array( [normal[0], normal[1], normal[2], d], dtype=np.float32 )
    return plane

def make_plane3( a, b, c, dtype=np.float32):
    """
    Form plane from three points that lie in the plane, NB the winding 
    order is anti-clockwise, ie right hand rule around a->b->c
    should point outwards
    """
    n = make_normal( a, b, c, dtype=dtype )  
    p = make_plane( n, a ) 
    return p 



def make_prism( angle, height, depth, dtype=np.float32, layout=0, crosscheck=True):
    #angle = 30
    #height = 200
    #depth = 200
    #dtype = np.float32
    #layout = 0 
    #crosscheck = True
    """
    Mid line of the symmetric prism spanning along z from -depth/2 to depth/2

                           v0
                            A  (0,height,0)     Y
                           /|\                  |
                          / | \                 |
                         /  |  \                +---- X
                        /   h   \              Z  
                       /    |    \ (x,y)   
                      M     |     N   
                     /      |      \
                    L-------O-------R   
                 

         (-hwidth,0, 0)           (hwidth, 0, 0)


    For apex angle 90 degrees, hwidth = height 
    """

    def a_(xyz, dtype=np.float32):
        a = np.asarray(xyz, dtype=dtype)
        a /= np.sqrt(np.dot(a,a))
        return a 


    hwidth = height*np.tan((np.pi/180.)*angle/2.) 
    v = np.zeros( (6,3), dtype=dtype)
    p = np.zeros( (5,4), dtype=dtype)

    if crosscheck:
        p2 = np.zeros( (5,4), dtype=dtype)
        n1 = np.zeros( (5,3), dtype=dtype)
        n2 = np.zeros( (5,3), dtype=dtype)


    if layout == 0: 
        """
        Front and back faces

                                    v3
                                     + 
                                    / \                     
                           v0      /   \                Y 
                            +     /     \               |
                           / \   /       \              | 
                          /   \ /         \             |
                         /     \           \            +---- X
                        /     / \           \          /  
                       /     /   \           \        /
                      /     +-----\-----------+      Z
                     /    v4       \          v5      
                    /               \
                   +-----------------+  
                  v1                 v2

                (x,y)

        """

        xmin = -hwidth
        xmax =  hwidth
        ymin = -height/2.
        ymax =  height/2.
        zmin = -depth/2.
        zmax =  depth/2.

        v[0] = [      0, ymax,  zmax ]   # front apex
        v[1] = [   xmin, ymin,  zmax ]   # front left
        v[2] = [   xmax, ymin,  zmax ]   # front right

        v[3] = [      0, ymax,  zmin ]   # back apex
        v[4] = [   xmin, ymin,  zmin ]   # back left
        v[5] = [   xmax, ymin,  zmin ]   # back right

        p[0] = make_plane3( v[5], v[3], v[0] )  
        p[1] = make_plane3( v[1], v[0], v[3] )
        p[2] = make_plane3( v[5], v[2], v[1] )
        p[3] = make_plane3( v[1], v[2], v[0] )
        p[4] = make_plane3( v[4], v[3], v[5] )

        if crosscheck:

            n2[0] = a_([ height, hwidth, 0])
            n2[1] = a_([-height, hwidth, 0])
            n2[2] = a_([ 0, -1,  0])
            n2[3] = a_([ 0,  0,  1])
            n2[4] = a_([ 0,  0, -1])

            # anti-clockwise from outside winding 
            n1[0] = make_normal( v[5], v[3], v[0] )  
            n1[1] = make_normal( v[1], v[0], v[3] )
            n1[2] = make_normal( v[5], v[2], v[1] )
            n1[3] = make_normal( v[1], v[2], v[0] )
            n1[4] = make_normal( v[4], v[3], v[5] )

            assert np.allclose(n1, n2)

            p2[0] = make_plane( n1[0], v[0] )  # +X+Y
            p2[1] = make_plane( n1[1], v[0] )  # -X+Y
            p2[2] = make_plane( n1[2], v[1] )  # -Y
            p2[3] = make_plane( n1[3], v[0] )  # +Z
            p2[4] = make_plane( n1[4], v[3] )  # -Z

            assert np.allclose(p, p2)
        pass



    elif layout == 1:
        """
        Front and back faces

                                    v3
                                     + 
                                    / \                     
                           v0      /   \                Z 
                            +     /     \               |  Y
                           / \   /       \              | /
                          /   \ /         \             |/
                         /     \           \            +---- X
                        /     / \           \            
                       /     /   \           \        
                      /     +-----\-----------+      
                     /    v4       \          v5      
                    /               \
                   +-----------------+  
                  v1                 v2

                (x,z)

        """

        xmin = -hwidth
        xmax =  hwidth
        ymin = -depth/2.
        ymax =  depth/2.
        zmin = -height/2.
        zmax =  height/2.

        v[0] = [       0,  ymin, zmax ]   # front apex
        v[1] = [    xmin,  ymin, zmin ]   # front left
        v[2] = [    xmax,  ymin, zmin ]   # front right

        v[3] = [       0,  ymax, zmax ]   # back apex
        v[4] = [    xmin,  ymax, zmin ]   # back left
        v[5] = [    xmax,  ymax, zmin ]   # back right

        p[0] = make_plane3( v[5], v[3], v[0] )  
        p[1] = make_plane3( v[1], v[0], v[3] )
        p[2] = make_plane3( v[5], v[2], v[1] )
        p[3] = make_plane3( v[1], v[2], v[0] )
        p[4] = make_plane3( v[4], v[3], v[5] )

        if crosscheck:

            n2[0] = a_([ height, 0, hwidth])  # +X+Z
            n2[1] = a_([-height, 0, hwidth])  # -X+Z
            n2[2] = a_([ 0, 0,-1])            # -Z
            n2[3] = a_([ 0,-1, 0])            # -Y
            n2[4] = a_([ 0, 1, 0])            # +Y

            # anti-clockwise from outside winding 
            n1[0] = make_normal( v[5], v[3], v[0] )  
            n1[1] = make_normal( v[1], v[0], v[3] )
            n1[2] = make_normal( v[5], v[2], v[1] )
            n1[3] = make_normal( v[1], v[2], v[0] )
            n1[4] = make_normal( v[4], v[3], v[5] )

            assert np.allclose(n1, n2)

            p2[0] = make_plane( n1[0], v[0] )  # +X+Z
            p2[1] = make_plane( n1[1], v[0] )  # -X+Z
            p2[2] = make_plane( n1[2], v[1] )  # -Z
            p2[3] = make_plane( n1[3], v[0] )  # -Y
            p2[4] = make_plane( n1[4], v[3] )  # +Y

            assert np.allclose(p, n2)
        pass     
    else:
        assert 0



    bbox = np.zeros( (2,3), dtype=dtype )
    for i in range(3):
        bbox[0,i] = np.min(v[:,i])
        bbox[1,i] = np.max(v[:,i])
    pass


    return p, v, bbox





def make_segment( phi0, phi1, sz, sr, dtype=np.float32 ):
    """

    Prism intended for deltaphi intersecting 

    :: 

               5 
              / \
             /   \
         sr /     \ 
           /       \
          /         \
         3-----------4       top plane at z = sz/2
              sr

 
               2 
              / \
             /   \
         sr /     \ 
           /       \
          /         \
         0-----------1        base plane at z = -sz/2
              sr   (x1,y1)


                                                     
                                       Z    
                                       |  Y
                                       | /
                                       |/
                                       +------ X
 
    """

    xy_ = lambda phi:np.array([np.cos(phi*np.pi/180.), np.sin(phi*np.pi/180.)], dtype=dtype)

    v = np.zeros( (6,3), dtype=dtype)   # verts

    x0,y0 = 0.,0. 
    x1,y1 = sr*xy_(phi0)
    x2,y2 = sr*xy_(phi1)
                                   
    v[0] = [    x0,     y0 , -sz/2. ]  
    v[1] = [    x1,     y1 , -sz/2. ]  
    v[2] = [    x2,     y2 , -sz/2. ]  

    v[3] = [    x0,     y0 ,  sz/2. ]  
    v[4] = [    x1,     y1 ,  sz/2. ]  
    v[5] = [    x2,     y2 ,  sz/2. ]  

    p = np.zeros( (5,4), dtype=dtype)   # planes
    p[0] = make_plane3( v[0], v[2], v[1] ) # -Z 
    p[1] = make_plane3( v[3], v[4], v[5] ) # +Z
    p[2] = make_plane3( v[0], v[1], v[3] ) # 
    p[3] = make_plane3( v[0], v[3], v[5] ) # 
    p[4] = make_plane3( v[2], v[5], v[4] ) # 

    b = np.zeros( (2,3), dtype=dtype )  # bbox
    for i in range(3):
        b[0,i] = np.min(v[:,i])
        b[1,i] = np.max(v[:,i])
    pass
    return p, v, b






def make_trapezoid( z, x1, y1, x2, y2, dtype=np.float32 ):
    """
    z-order verts


                  6----------7
                 /|         /|
                / |        / |
               4----------5  |
               |  |       |  |                       
               |  |       |  |         Z    
               |  2-------|--3         |  Y
               | /        | /          | /
               |/         |/           |/
               0----------1            +------ X
                         

    x1: x length at -z
    y1: y length at -z

    x2: x length at +z
    y2: y length at +z

    z:  z length

    """ 
    v = np.zeros( (8,3), dtype=dtype)   # verts
                                    # ZYX
    v[0] = [ -x1/2., -y1/2. , -z/2. ]  # 000
    v[1] = [  x1/2., -y1/2. , -z/2. ]  # 001 
    v[2] = [ -x1/2.,  y1/2. , -z/2. ]  # 010
    v[3] = [  x1/2.,  y1/2. , -z/2. ]  # 011

    v[4] = [ -x2/2., -y2/2. ,  z/2. ]  # 100
    v[5] = [  x2/2., -y2/2. ,  z/2. ]  # 101
    v[6] = [ -x2/2.,  y2/2. ,  z/2. ]  # 110
    v[7] = [  x2/2.,  y2/2. ,  z/2. ]  # 111

    p = np.zeros( (6,4), dtype=dtype)   # planes
    p[0] = make_plane3( v[3], v[7], v[5] ) # +X  
    p[1] = make_plane3( v[0], v[4], v[6] ) # -X
    p[2] = make_plane3( v[2], v[6], v[7] ) # +Y
    p[3] = make_plane3( v[1], v[5], v[4] ) # -Y
    p[4] = make_plane3( v[5], v[7], v[6] ) # +Z
    p[5] = make_plane3( v[3], v[1], v[0] ) # -Z

    b = np.zeros( (2,3), dtype=dtype )  # bbox
    for i in range(3):
        b[0,i] = np.min(v[:,i])
        b[1,i] = np.max(v[:,i])
    pass
    return p, v, b



def make_icosahedron(scale=500., dtype=np.float32):
    """
    dtype = np.float32

    * 20 faces (triangular)
    * 12 verts
    * 30 edges

    Translation of npy-/NTrianglesNPY::icosahedron()

    """
    CZ = 2./np.sqrt(5)
    SZ = 1./np.sqrt(5)

    C1 = np.cos( np.pi*18./180. )
    S1 = np.sin( np.pi*18./180. )
    C2 = np.cos( np.pi*54./180. )
    S2 = np.sin( np.pi*54./180. )

    SC = 1.
    X1 = C1*CZ
    Y1 = S1*CZ
    X2 = C2*CZ
    Y2 = S2*CZ

    SC *= scale
    X2 *= scale
    Y2 *= scale
    SZ *= scale
    X1 *= scale
    Y1 *= scale
    CZ *= scale

    vec3_ = lambda x,y,z:np.array([x,y,z], dtype=dtype) 

    Ip0 = vec3_(0,0,SC) 
    Ip1 = vec3_(-X2,-Y2,SZ) 
    Ip2 = vec3_( X2,-Y2,SZ) 
    Ip3 = vec3_( X1, Y1,SZ) 
    Ip4 = vec3_(  0, CZ,SZ) 
    Ip5 = vec3_(-X1, Y1,SZ) 

    Im0 = vec3_(-X1, -Y1,-SZ) 
    Im1 = vec3_(  0, -CZ,-SZ) 
    Im2 = vec3_( X1, -Y1,-SZ) 
    Im3 = vec3_( X2,  Y2,-SZ) 
    Im4 = vec3_(-X2,  Y2,-SZ) 
    Im5 = vec3_(0,0,-SC) 

    v = np.hstack( [Ip0,Ip1,Ip2,Ip3,Ip4,Ip5, Im0,Im1,Im2,Im3,Im4,Im5] ).reshape(-1,3)
    p = np.zeros( (20,4), dtype=dtype)   # planes

    # front pole 
    p[ 0] = make_plane3(Ip0, Ip1, Ip2)
    p[ 1] = make_plane3(Ip0, Ip5, Ip1)
    p[ 2] = make_plane3(Ip0, Ip4, Ip5)
    p[ 3] = make_plane3(Ip0, Ip3, Ip4)
    p[ 4] = make_plane3(Ip0, Ip2, Ip3)

    # mid 
    p[ 5] = make_plane3(Ip1, Im0, Im1)
    p[ 6] = make_plane3(Im0, Ip1, Ip5)
    p[ 7] = make_plane3(Ip5, Im4, Im0)
    p[ 8] = make_plane3(Im4, Ip5, Ip4)
    p[ 9] = make_plane3(Ip4, Im3, Im4)
    p[10] = make_plane3(Im3, Ip4, Ip3)
    p[11] = make_plane3(Ip3, Im2, Im3)
    p[12] = make_plane3(Im2, Ip3, Ip2)
    p[13] = make_plane3(Ip2, Im1, Im2)
    p[14] = make_plane3(Im1, Ip2, Ip1)

    # back pole
    p[15] = make_plane3(Im3, Im2, Im5)
    p[16] = make_plane3(Im4, Im3, Im5)
    p[17] = make_plane3(Im0, Im4, Im5)
    p[18] = make_plane3(Im1, Im0, Im5)
    p[19] = make_plane3(Im2, Im1, Im5)

    b = np.zeros( (2,3), dtype=dtype )  # bbox
    for i in range(3):
        b[0,i] = np.min(v[:,i])
        b[1,i] = np.max(v[:,i])
    pass
    return p, v, b




if __name__ == '__main__':
    #p, v, b = make_prism( 45, 400,  400 )
    #p, v, b = make_trapezoid(z=50.02, x1=100, y1=27, x2=237.2, y2=27 )
    #p, v, b = make_icosahedron()
    p, v, b = make_segment(0,45,100,200)
    pass


