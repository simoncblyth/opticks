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



def make_prism( angle, height, depth, dtype=np.float32, layout=0):
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
    p = np.zeros( (6,4), dtype=dtype)   # planes
    b = np.zeros( (2,3), dtype=dtype )  # bbox

                                    # ZYX
    v[0] = [ -x1/2., -y1/2. , -z ]  # 000
    v[1] = [  x1/2., -y1/2. , -z ]  # 001 
    v[2] = [ -x1/2.,  y1/2. , -z ]  # 010
    v[3] = [  x1/2.,  y1/2. , -z ]  # 011

    v[4] = [ -x2/2., -y2/2. ,  z ]  # 100
    v[5] = [  x2/2., -y2/2. ,  z ]  # 101
    v[6] = [ -x2/2.,  y2/2. ,  z ]  # 110
    v[7] = [  x2/2.,  y2/2. ,  z ]  # 111

    p[0] = make_plane3( v[3], v[7], v[5] ) # +X  
    p[1] = make_plane3( v[0], v[4], v[6] ) # -X
    p[2] = make_plane3( v[2], v[6], v[7] ) # +Y
    p[3] = make_plane3( v[1], v[5], v[4] ) # -Y
    p[4] = make_plane3( v[5], v[7], v[6] ) # +Z
    p[5] = make_plane3( v[3], v[1], v[0] ) # -Z

    for i in range(3):
        b[0,i] = np.min(v[:,i])
        b[1,i] = np.max(v[:,i])
    pass
    return p, v, b









if __name__ == '__main__':
    #p, v, bbox = make_prism( 45, 400,  400 )
    p, v, bbox = make_trapezoid(z=50.02, x1=100, y1=27, x2=237.2, y2=27 )


