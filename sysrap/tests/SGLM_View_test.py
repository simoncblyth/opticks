#!/usr/bin/env python
"""

~/o/sysrap/tests/SGLM_View_test.sh prep

"""
import os
import numpy as np
np.set_printoptions(suppress=True)





def make_demo():
     vv = np.zeros( [4, 4, 4], dtype=np.float32  )
     uu = vv.view(np.uint32)

     # EYE
     vv[0][0] = [-10,1,0,1]
     vv[1][0] = [-5 ,2,0,1]
     vv[2][0] = [+5 ,3,0,1]
     vv[3][0] = [+10,4,0,1]

     # LOOK
     vv[0][1] = [0,0,0,1]
     vv[1][1] = [0,0,0,1]
     vv[2][1] = [0,0,0,1]
     vv[3][1] = [0,0,0,1]

     # UP
     vv[0][2] = [0,0,1,0]
     vv[1][2] = [0,0,1,0]
     vv[2][2] = [0,0,1,0]
     vv[3][2] = [0,0,1,0]

     # CTRL
     uu[0][3] = [0,0,0,0]
     uu[1][3] = [1,1,1,1]
     uu[2][3] = [2,2,2,2]
     uu[3][3] = [3,3,3,3]

     return vv


def make_circle_looking_inwards(n=24):
     vv = np.zeros( [n, 4, 4], dtype=np.float32  )
     uu = vv.view(np.uint32)
     for i in range(n):
         phi = 2*np.pi*i/n
         sp = np.sin(phi)
         cp = np.cos(phi)
         vv[i][0] = [sp,cp,0,1]  # EYE
         vv[i][1] = [0,0,0,1]    # LOOK
         vv[i][2] = [0,0,1,0]    # UP
         uu[i][3] = [0,1,2,3]    # CTRL (currently unused)
     pass
     return vv

def make_circle_looking_tangential(n=24,z=0,r=1):
     vv = np.zeros( [n, 4, 4], dtype=np.float32  )
     uu = vv.view(np.uint32)
     for i in range(n):
         phi = 2*np.pi*i/n
         sp = np.sin(phi)
         cp = np.cos(phi)
         delta = np.pi/2
         sp1 = np.sin(phi+delta)
         cp1 = np.cos(phi+delta)

         vv[i][0] = [r*cp,r*sp,z,1]  # EYE
         vv[i][1] = [r*cp1,r*sp1,z,1]    # LOOK
         vv[i][2] = [0,0,1,0]    # UP
         uu[i][3] = [0,1,2,3]    # CTRL (currently unused)
     pass
     return vv


def make_multicircle_looking_tangential(nn=[24,24,24],zz=[-1,0,1], rr=[1,1,1]):
    assert len(zz) == len(rr)
    assert len(zz) == len(nn)
    num = len(zz)
    l = []
    for i in range(num):
        l.append(make_circle_looking_tangential(n=nn[i], z=zz[i], r=rr[i]))
    pass
    return np.concat(tuple(l))



def create_model_to_world(ax, up, translation):
    """


            z
            |
            |
            |
            |
            +--------- y
           /
          /
         x

    # transpose is lazy, without np.ascontiguousarray get unexpected fortran_order True
    """
    right = np.cross(up, ax)
    right /= np.linalg.norm(right)
    up_rectified = np.cross(ax, right)
    m2w = np.eye(4)
    m2w[:3, 0] = right       # X-axis
    m2w[:3, 1] = up_rectified # Y-axis
    m2w[:3, 2] = ax           # Z-axis
    m2w[:3, 3] = translation
    return np.ascontiguousarray(m2w.T)



class EMFTiltedRings:
    theta = 56. * np.pi / 180.
    phi = -54. * np.pi / 180.

    zc = np.fromstring("""
            21.271,  20.602,  19.554,  18.555,
            17.077,  15.600,  14.122,  12.644,
            11.166,   9.685,   8.207,   6.729,
             5.251,   3.773,   2.295,   0.817,
            -0.821,  -2.299,  -3.777,  -5.255,
            -6.733,  -8.210,  -9.688, -11.166,
           -12.644, -14.122, -15.600, -17.077,
           -18.555, -19.554, -20.602, -21.271
            """, sep="," )


    rc = np.fromstring("""
            3.843,   6.509,   9.185,  11.053,
           13.210,  14.924,  16.325,  17.495,
           18.465,  19.282,  19.951,  20.502,
           20.930,  21.240,  21.450,  21.559,
           21.559,  21.450,  21.240,  20.930,
           20.502,  19.951,  19.282,  18.465,
           17.495,  16.325,  14.924,  13.210,
           11.053,   9.185,   6.509,   3.843
           """, sep="," )

    L_leg_mm  = 56.0;
    t_mm      = 4.0;
    holder_center_offset_mm = L_leg_mm*0.5 - t_mm   # Offset from "web-bottom" to geometric center along the tilted axis.

    def __init__(self):
        pass

    def __call__(self):
        assert len(self.rc) == len(self.zc)
        num = len(self.rc)

        st, ct = np.sin(self.theta), np.cos(self.theta)
        sp, cp = np.sin(self.phi), np.cos(self.phi)
        ax = np.array([st * cp, st * sp, ct])

        world_z = np.array([0, 0, 1])
        up = world_z - np.dot(world_z, ax) * ax
        up /= np.linalg.norm(up)
        tr = np.array([0,0,0])

        m2w = create_model_to_world(ax, up, tr )

        nn = np.repeat(24, num)
        zz = self.zc*1000/20000
        rr = self.rc*1000/20000
        vv = make_multicircle_looking_tangential(nn=nn, zz=zz, rr=rr )

        return vv, m2w

def make_emf_tilted_rings():
    etr  = EMFTiltedRings()
    vv0, m2w = etr()
    vv = np.dot( vv0, m2w )
    return vv0



if __name__ == '__main__':

     #vv = make_demo()
     #vv = make_circle_looking_inwards()
     #vv = make_circle_looking_tangential()
     #vv = make_multicircle_looking_tangential()
     vv = make_emf_tilted_rings()

     if "FOLD" in os.environ:
         outpath = os.path.expandvars("$FOLD/SGLM_View_test.npy")
         print(f"save vv {vv.shape} to {outpath}")
         np.save(outpath, vv )
     else:
         print("define FOLD envvar to save")
     pass




