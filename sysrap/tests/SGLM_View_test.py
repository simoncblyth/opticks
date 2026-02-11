#!/usr/bin/env python

import os
import numpy as np


if __name__ == '__main__':

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

     outpath = os.path.expandvars("$FOLD/SGLM_View_test.npy")
     print(f"save to {outpath}")
     np.save(outpath, vv)





