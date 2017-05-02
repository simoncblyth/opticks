#!/usr/bin/env python
"""
"""
import numpy as np
fromstring_  = lambda s:np.fromstring(s, dtype=np.float32, sep=",") 



def scale(arg=[1,2,3], m=None, dtype=np.float32):
    """  
    Translation of glm::scale into numpy 
    /usr/local/opticks/externals/glm/glm-0.9.6.3/glm/gtc/matrix_transform.inl
    """
    if m is None:m = np.eye(4, dtype=dtype)

    if type(arg) is str:
        arg = fromstring_(arg)
        assert arg.shape == (3,)
    elif arg is None:
        arg = [1,1,1]
    else:
        pass
    pass
    v = np.asarray(arg, dtype=dtype)

    Result = np.eye(4, dtype=dtype)
    Result[0] = m[0] * v[0];
    Result[1] = m[1] * v[1];
    Result[2] = m[2] * v[2];
    Result[3] = m[3];
    return Result

def translate(arg=[1,2,3], m=None, dtype=np.float32):
    """  
    Translation of glm::translate into numpy 
    /usr/local/opticks/externals/glm/glm-0.9.6.3/glm/gtc/matrix_transform.inl
    """
    if type(arg) is str:
        arg = fromstring_(arg)
        assert arg.shape == (3,)
    elif arg is None:
        arg = [0,0,0]
    else:
        pass
    pass
    v = np.asarray(arg, dtype=dtype)

    if m is None:m = np.eye(4, dtype=dtype)
    Result = np.eye(4, dtype=dtype)
    Result[3] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3]
    return Result


def rotate_three_axis(arg=[0,0,90], m=None, dtype=np.float32):
    """
    :param arg: 3-component array 
    """
    if m is None:m = np.eye(4, dtype=dtype)

    if arg is not None:
        assert len(arg) == 3, arg
        rx = arg[0]
        ry = arg[1]
        rz = arg[2]

        if rx != 0: m = rotate([1,0,0,rx], m )
        if ry != 0: m = rotate([0,1,0,ry], m )
        if rz != 0: m = rotate([0,0,1,rz], m )
    pass

    return m 
   


def rotate(arg=[0,0,1,45], m=None, dtype=np.float32):
    """
    :param arg: 4-component array, 1st three for axis, 4th for rotation angle in degrees
    :param m: optional matrix to combine with  

    Translation of glm::rotate into numpy 
    /usr/local/opticks/externals/glm/glm-0.9.6.3/glm/gtc/matrix_transform.inl::
    """
    if m is None:m = np.eye(4, dtype=dtype)

    if type(arg) is str:
        arg = fromstring_(arg)
        assert arg.shape == (4,)
    elif arg is None:
        arg = [0,0,1,0]
    else:
        pass
    pass
    v = np.asarray(arg, dtype=dtype)

    axis_ = v[:3] 
    angle_d = v[3]

    axis = np.array( axis_, dtype=dtype)
    axis /= np.sqrt(np.dot(axis,axis))

    angle = np.pi*float(angle_d)/180.
    c = np.cos(angle)
    s = np.sin(angle)

    temp = (1. - c)*axis

    Rotate  = np.eye(3, dtype=dtype)
    Rotate[0][0] = c + temp[0] * axis[0]
    Rotate[0][1] = 0 + temp[0] * axis[1] + s * axis[2]
    Rotate[0][2] = 0 + temp[0] * axis[2] - s * axis[1]
    
    Rotate[1][0] = 0 + temp[1] * axis[0] - s * axis[2]
    Rotate[1][1] = c + temp[1] * axis[1]
    Rotate[1][2] = 0 + temp[1] * axis[2] + s * axis[0]
    
    Rotate[2][0] = 0 + temp[2] * axis[0] + s * axis[1]
    Rotate[2][1] = 0 + temp[2] * axis[1] - s * axis[0]
    Rotate[2][2] = c + temp[2] * axis[2]
 
    Result = np.eye(4, dtype=dtype)
    Result[0] = m[0] * Rotate[0][0] + m[1] * Rotate[0][1] + m[2] * Rotate[0][2];
    Result[1] = m[0] * Rotate[1][0] + m[1] * Rotate[1][1] + m[2] * Rotate[1][2];
    Result[2] = m[0] * Rotate[2][0] + m[1] * Rotate[2][1] + m[2] * Rotate[2][2];
    Result[3] = m[3];

    return Result;




def make_transform( order, tla, rot, sca, dtype=np.float32, suppress_identity=True, three_axis_rotate=False):
    """
    :param order: string containing "s" "r" and "t", standard order is "trs" meaning t*r*s  ie scale first, then rotate, then translate 
    :param tla: tx,ty,tz tranlation dists eg 0,0,0 for no translation 
    :param rot: ax,ay,az,angle_degrees  eg 0,0,1,45 for 45 degrees about z-axis
    :param sca: sx,sy,sz eg 1,1,1 for no scaling 
    :return mat: 4x4 numpy array 

    All arguments can be specified as comma delimited string, list or numpy array

    Translation of npy/tests/NGLMTest.cc:make_mat
    """

    if tla is None and rot is None and sca is None:
        return None

    identity = np.eye(4, dtype=dtype)
    m = np.eye(4, dtype=dtype) 
    for c in order:
        if c == 's':
            m = scale(sca, m)
        elif c == 'r':
            if three_axis_rotate:
                m = rotate_three_axis(rot, m)
            else:
                m = rotate(rot, m)
            pass
        elif c == 't':
            m = translate(tla, m)
        else:
            assert 0
        pass
    pass

    if suppress_identity and np.all( m == identity ):
        return None
    pass
    return m 



def make_trs( tla, rot, sca, three_axis_rotate=False, dtype=np.float32):
    return make_transform("trs", tla, rot, sca, three_axis_rotate=three_axis_rotate, dtype=dtype ) 


def test_make_transform():
    """
    compare with npy/tests/NGLMTest.cc:test_make_transform
    """
    fromstr = True
    if fromstr:
        tla = "0,0,100"
        rot = "0,0,1,45"
        sca = "1,2,3"
    else: 
        tla = [0,0,100]
        rot = [0,0,1,45]
        sca = [1,2,3]
    pass

    dtype = np.float32
    t = make_transform("t", tla, rot, sca, dtype=dtype )
    assert(t.dtype == dtype)

    r = make_transform("r", tla, rot, sca, dtype=dtype)
    assert(t.dtype == dtype)

    s = make_transform("s", tla, rot, sca, dtype=dtype)
    assert(s.dtype == dtype)

    trs = make_transform("trs", tla, rot, sca, dtype=dtype)
    assert(trs.dtype == dtype)

    trs2 = make_trs(tla, rot, sca, dtype=dtype)
    assert(trs2.dtype == dtype )


    print "t\n", t
    print "r\n", r
    print "s\n", s
    print "trs\n", trs
   



if __name__ == '__main__':

    rot = rotate([0,0,1,45])
    print "rot\n", rot 

    sca = scale([1,2,3])
    print "sca\n", sca

    test_make_transform()

    rot3x = rotate_three_axis([45,0,0])
    rot3y = rotate_three_axis([0,45,0])
    rot3z = rotate_three_axis([0,0,45])

    rot3xyz = rotate_three_axis([45,45,45])


    print "rot3x\n", rot3x
    print "rot3y\n", rot3y
    print "rot3z\n", rot3z

    print "rot3xyz\n", rot3xyz





