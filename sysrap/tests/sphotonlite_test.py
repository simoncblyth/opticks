#!/usr/bin/env python
import numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.sphotonlite import SPhotonLite

if __name__ == "__main__":
    f = Fold.Load(symbol="f")                 
    p = SPhotonLite.view(f.demoarray)  # demoarray shape (N,4) uint32

    N = len(p)
    x_hitcount = np.arange(N, dtype=np.uint16)
    x_identity = np.arange(0, N*1000, 1000, dtype=np.uint16)
    x_time     = np.arange(0, N*0.1, 0.1, dtype=np.float32)
    x_lposcost = np.full(N, 0.5, dtype=np.float32)
    x_lposfphi = np.full(N, 0.6, dtype=np.float32)
    x_flagmask = np.full(N, 0x2040, dtype=np.uint32)

    atol_32 = 1e-10
    atol_16 = 1e-5

    assert np.all(x_hitcount == p["hitcount"] )
    assert np.all(x_identity == p["identity"] )
    assert np.allclose(x_time, p["time"] , atol=atol_32 )
    assert np.allclose(x_lposcost, p["lposcost_f"], atol=atol_16)
    assert np.allclose(x_lposfphi, p["lposfphi_f"], atol=atol_16)
    assert np.all(x_flagmask == p["flagmask"] )

    print("All checks passed – photonlite data are now a tidy record array!")


