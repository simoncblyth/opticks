#!/usr/bin/env python
"""
"""
import numpy as np


def make_plane( normal, point, dtype=np.float32 ):

    normal = np.asarray(normal, dtype=dtype)
    point = np.asarray(point, dtype=dtype)

    assert normal.shape == (3,)
    assert point.shape == (3,)
    normal /= np.sqrt(np.dot(normal,normal))

    d = np.dot(normal, point)  # distance of point from origin 
    plane = np.array( [normal[0], normal[1], normal[2], d], dtype=np.float32 )
    return plane


def make_prism( angle, height, depth, dtype=np.float32, layout=1):
    """
    Mid line of the symmetric prism spanning along z from -depth/2 to depth/2

                                                 
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

    a_ = lambda _:np.array(_, dtype=dtype)
    hwidth = height*np.tan((np.pi/180.)*angle/2.) 
    verts = np.zeros( (6,3), dtype=dtype)
    planes = np.zeros( (5,4), dtype=dtype)
    # form planes from their normals and any point that lies on the plane   

    if layout == 0: 

        ymax =  height/2.
        ymin =  -height/2.

        verts[0] = [       0, ymax,  depth/2. ]   # front apex
        verts[1] = [ -hwidth, ymin,  depth/2. ]   # front left
        verts[2] = [  hwidth, ymin,  depth/2. ]   # front right
        verts[3] = [       0, ymax, -depth/2. ]   # back apex
        verts[4] = [ -hwidth, ymin, -depth/2. ]   # back left
        verts[5] = [  hwidth, ymin, -depth/2. ]   # back right

        front_apex = verts[0]
        back_apex = verts[3]
        front_left = verts[1] 

        planes[0] = make_plane( a_([ height, hwidth, 0]),  front_apex )  # +X+Y
        planes[1] = make_plane( a_([-height, hwidth, 0]),  front_apex )  # -X+Y
        planes[2] = make_plane( a_([ 0,          -1,  0]), front_left )  # -Y
        planes[3] = make_plane( a_([ 0,           0,  1]), front_apex)  # +Z
        planes[4] = make_plane( a_([ 0,           0, -1]), back_apex )  # -Z

    elif layout == 1:
        """
        Midline Y=0 triangle of the prism

                                                Z    
                            A  (0,0,zmax)       |  Y
                           /|\                  | /
                          / | \                 |/
                         /  |  \                +---- X
                        /   |   \                
                       /    |    \ (x,z)   
                      /     |     \   
                     /      |      \
                  L +-------+-------+  R 
           (-hwidth,0, zmin)        (hwidth, 0, zmin)

                  
        """
        zmax =  height/2.
        zmin =  -height/2.

        verts[0] = [       0, -depth/2., zmax ]   # front apex
        verts[1] = [ -hwidth, -depth/2., zmin ]   # front left
        verts[2] = [  hwidth, -depth/2., zmin ]   # front right
        verts[3] = [       0,  depth/2., zmax ]   # back apex
        verts[4] = [ -hwidth,  depth/2., zmin ]   # back left
        verts[5] = [  hwidth,  depth/2., zmin ]   # back right

        front_apex = verts[0]   # -Y
        back_apex = verts[3]    # +Y
        front_left = verts[1]   # -Y

        planes[0] = make_plane( a_([ height, 0, hwidth]),  front_apex )  # +X+Z
        planes[1] = make_plane( a_([-height, 0, hwidth]),  front_apex )  # -X+Z
        planes[2] = make_plane( a_([ 0, 0,-1]), front_left )  # -Z
        planes[3] = make_plane( a_([ 0,-1, 0]), front_apex)   # -Y
        planes[4] = make_plane( a_([ 0, 1, 0]), back_apex )   # +Y

    else:
        assert 0



    bbox = np.zeros( (2,3), dtype=dtype )
    for i in range(3):
        bbox[0,i] = np.min(verts[:,i])
        bbox[1,i] = np.max(verts[:,i])
    pass


    return planes, verts, bbox


if __name__ == '__main__':



    planes, verts, bbox = make_prism( 45, 400,  400 )


