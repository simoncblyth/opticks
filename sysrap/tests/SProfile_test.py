#!/usr/bin/env python

import numpy as np

if __name__ == '__main__':

    base = os.path.expandvars("$FOLD")
    name = "SProfile.npy"
    evts =  sorted(map(int,filter(lambda _:_.isdigit(), os.listdir(base))))
 
    ee = {}   
    for evt in evts:
        path = os.path.join(base, "%s"%evt, name)
        if not os.path.exists(path): continue
        ee[evt] = np.load(path)
    pass

    for evt in evts:
        a = ee[evt]
        print("ee[%d] %s " % (evt, str(a.shape)) )

        aa_ = "a[:,1:]-a[0,1]" 
        aa = eval(aa_)
        print(aa_,repr(aa))

        daa_ = "np.diff(aa.reshape(-1))"
        daa = eval(daa_)
        print(daa_,repr(daa))
    pass


pass

