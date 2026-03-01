#!/usr/bin/env python
"""

~/o/sysrap/tests/SGLM_View_test.sh prep

"""
import os
import numpy as np

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

def make_circle_looking_tangential(n=24):
     vv = np.zeros( [n, 4, 4], dtype=np.float32  )
     uu = vv.view(np.uint32)
     for i in range(n):
         phi = 2*np.pi*i/n
         sp = np.sin(phi)
         cp = np.cos(phi)
         delta = np.pi/2
         sp1 = np.sin(phi+delta)
         cp1 = np.cos(phi+delta)

         vv[i][0] = [cp,sp,0,1]  # EYE
         vv[i][1] = [cp1,sp1,0,1]    # LOOK
         vv[i][2] = [0,0,1,0]    # UP
         uu[i][3] = [0,1,2,3]    # CTRL (currently unused)
     pass
     return vv



if __name__ == '__main__':

     #vv = make_demo()
     #vv = make_circle_looking_inwards()
     vv = make_circle_looking_tangential()

     outpath = os.path.expandvars("$FOLD/SGLM_View_test.npy")
     print(f"save vv {vv.shape} to {outpath}")
     np.save(outpath, vv )





