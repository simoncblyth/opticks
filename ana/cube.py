#!/usr/bin/env python
"""
See cubeplt.py for 3d plotting of the cubes
"""
import numpy as np

def make_pyvista_indices(indices):
    """
    :param indices:  (nface,3) triangles OR (nface,4) quads 
    :return ii: vista type list 
    """
    sh = list(indices.shape)   
    last = sh[-1]
    assert last in (3,4)
    sh[-1] = last+1
    ii = np.zeros(sh, dtype=np.int32) 
    ii[:,1:] = indices 
    ii[:,0] = last
    return ii  


def make_cube_oxyz(oxyz):
    """
    :param oxyz: (4,3) : origin and orthogonal surrounding points nominally 
                         assumed in +X,+Y,+Z directions relative to first point 


              (YZ)      (XYZ)
               6.........7 
              /.        /|
             / .       / | 
            /  .      /  |
           3.........5   |
           |   .     |   |
           |   2.....|...4
         Z |  /      |  /
           | / Y     | /
           |/        |/
           0---------1
                X

    """
    o = oxyz[0]
    v = oxyz[1:] - o   # three : assumed orthogonal base vectors   

    verts = np.zeros([8,3], dtype=np.float32) 
    verts[:4] = oxyz
    verts[4] = o + v[0] + v[1]        # XY
    verts[5] = o + v[0] + v[2]        # XZ
    verts[6] = o + v[1] + v[2]        # YZ
    verts[7] = o + v[0] + v[1] + v[2] # XYZ

    indices = np.array([
           [0,1,5,3],    # thumb-out   (outwards normal)
           [1,4,7,5],    # thumb-right (outwards normal)
           [4,2,6,7],    # thumb-back  (outwards normal)
           [0,3,6,2],    # thumb-left  (outwards normal) 
           [0,2,4,1],    # thumb-down  (outwards normal)
           [5,7,6,3]],   # thumb-up    (outwards normal)  
           dtype=np.int32)
    
    pv_indices = make_pyvista_indices(indices)
    faces = verts[indices]
    assert verts.shape == (8,3)
    assert indices.shape == (6,4)
    assert pv_indices.shape == (6,5)
    assert faces.shape == (6,4,3)
    return verts, faces, pv_indices


def make_cube_bbox(bbox):
    """
    :param bbox: (2,3)  mi,mx 

                         mx    
              (YZ)      (XYZ)
               6.........7 
              /.        /|
             / .       / | 
            /  .      /  |
           3.........5   |
           |   .     |   |
           |   2.....|...4
         Z |  /      |  /
           | / Y     | /
           |/        |/
           0---------1
         mi     X

    """
    assert bbox.shape == (2,3)
    mi, mx = bbox
    verts = np.zeros([8,3], dtype=np.float32) 
    verts[0] = [mi[0], mi[1], mi[2]]  # ---
    verts[1] = [mx[0], mi[1], mi[2]]  # X--
    verts[2] = [mi[0], mx[1], mi[2]]  # -Y-
    verts[3] = [mi[0], mi[1], mx[2]]  # --Z
    verts[4] = [mx[0], mx[1], mi[2]]  # XY- 
    verts[5] = [mx[0], mi[1], mx[2]]  # X-Z
    verts[6] = [mi[0], mx[1], mx[2]]  # -YZ 
    verts[7] = [mx[0], mx[1], mx[2]]  # XYZ 

    indices = np.array([
           [0,1,5,3],    # thumb-out   (outwards normal)
           [1,4,7,5],    # thumb-right (outwards normal)
           [4,2,6,7],    # thumb-back  (outwards normal)
           [0,3,6,2],    # thumb-left  (outwards normal) 
           [0,2,4,1],    # thumb-down  (outwards normal)
           [5,7,6,3]],   # thumb-up    (outwards normal)  
           dtype=np.int32)
    pv_indices = make_pyvista_indices(indices)
    
    faces = verts[indices]
    assert verts.shape == (8,3)
    assert indices.shape == (6,4)
    assert pv_indices.shape == (6,5)
    assert faces.shape == (6,4,3)

    return verts, faces, pv_indices

    
def make_cube(cube):
    """
    :param cube: shape of either (4,3) or (2,3) 
    """
    if cube.shape == (4,3):
        return make_cube_oxyz(cube)
    elif cube.shape == (2,3):
        return make_cube_bbox(cube)
    else:
        assert 0, ("cube specification array not handled", cube.shape)
    pass


if __name__ == '__main__':

    oxyz = np.array([(0,0,0), (100,0,0), (0,100,0), (0,0,100)], dtype=np.float32)
    bbox = np.array([(0,0,0),(100,100,100)], dtype=np.float32)

    points0, faces0, indices0 = make_cube(oxyz)
    points1, faces1, indices1 = make_cube(bbox)

    assert np.all( points0 == points1 ) 
    assert np.all( faces0 == faces1 ) 




