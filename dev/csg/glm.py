#!/usr/bin/env python
"""
"""
import numpy as np

def rotate(arg=[0,0,1,45], m=None ):
    """
    :param arg: 4-component array, 1st three for axis, 4th for rotation angle in degrees
    :param m: optional matrix to combine with  

    Translation of glm::rotate into numpy 
    /usr/local/opticks/externals/glm/glm-0.9.6.3/glm/gtc/matrix_transform.inl::
    """
    if m is None:m = np.eye(4)

    arg = np.asarray(arg, dtype=np.float32)
    axis_ = arg[:3] 
    angle_d = arg[3]

    axis = np.array( axis_, dtype=np.float32)
    axis /= np.sqrt(np.dot(axis,axis))

    angle = np.pi*float(angle_d)/180.
    c = np.cos(angle)
    s = np.sin(angle)

    temp = (1. - c)*axis

    Rotate  = np.eye(3)
    Rotate[0][0] = c + temp[0] * axis[0]
    Rotate[0][1] = 0 + temp[0] * axis[1] + s * axis[2]
    Rotate[0][2] = 0 + temp[0] * axis[2] - s * axis[1]
    
    Rotate[1][0] = 0 + temp[1] * axis[0] - s * axis[2]
    Rotate[1][1] = c + temp[1] * axis[1]
    Rotate[1][2] = 0 + temp[1] * axis[2] + s * axis[0]
    
    Rotate[2][0] = 0 + temp[2] * axis[0] + s * axis[1]
    Rotate[2][1] = 0 + temp[2] * axis[1] - s * axis[0]
    Rotate[2][2] = c + temp[2] * axis[2]
 
    Result = np.eye(4)
    Result[0] = m[0] * Rotate[0][0] + m[1] * Rotate[0][1] + m[2] * Rotate[0][2];
    Result[1] = m[0] * Rotate[1][0] + m[1] * Rotate[1][1] + m[2] * Rotate[1][2];
    Result[2] = m[0] * Rotate[2][0] + m[1] * Rotate[2][1] + m[2] * Rotate[2][2];
    Result[3] = m[3];

    return Result;


if __name__ == '__main__':

    rot = rotate([0,0,1,45])
    print rot 











