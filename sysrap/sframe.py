#!/usr/bin/env python

from opticks.ana.npmeta import NPMeta
import numpy as np

class sframe(object):
    @classmethod
    def Load(cls, fold=None, name="sframe.npy"):
        if fold is None:
            fold = os.environ.get("FOLD", "")
        pass
        path = os.path.join(fold, name)
        return cls(path)

    def __init__(self, path, clear_identity=True ):
        """
        Whether clear_identity makes any material difference depends on the identity values. 
        But it should be done anyhow.  For some identity values they will appear as nan in float.
        """

        metapath = path.replace(".npy", "_meta.txt")
        if os.path.exists(metapath):
            mtxt = np.loadtxt(metapath, dtype="|S100", delimiter="\t" )
            meta = NPMeta(mtxt)
        else:
            mtxt = None
            meta = None
        pass   
        self.mtxt = mtxt
        self.meta = meta 


        a = np.load(path)
        i = a.view(np.int32)

        self.path = path 
        self.a = a
        self.i = i

        ce = a[0,0]
        ix0,ix1,iy0,iy1 = i[0,1,:4]        # q1.i.xyzw
        iz0,iz1,num_photon = i[0,2,:3]     # q2.i.xyz  
        gridscale = a[0,2,3]               # q2.f.w

        midx, mord, iidx = i[0,3,:3]       # q3.i.xyz
        inst = i[0,3,3]     # q3.i.w

        self.midx = midx
        self.mord = mord
        self.iidx = iidx

        self.inst = inst


        grid = "".join(["ix0 %(ix0)4d ix1 %(ix1)4d ",
                        "iy0 %(iy0)4d iy1 %(iy1)4d ",
                        "iz0 %(iz0)4d iz1 %(iz1)4d ",
                        "num_photon %(num_photon)4d gridscale %(gridscale)10.4f"]) % locals() 

        self.grid = grid

        self.ce = ce 
        self.ix0 = ix0
        self.ix1 = ix1
        self.iy0 = iy0
        self.iy1 = iy1
        self.iz0 = iz0
        self.iz1 = iz1
        self.num_photon = num_photon
        self.gridscale = gridscale


        target = "midx %(midx)6d mord %(mord)6d iidx %(iidx)6d       inst %(inst)7d   " % locals() 
        self.target = target 


        ins_idx_1 = i[1,0,3] - 1   # q0.u.w-1  see sqat4.h qat4::getIdentity qat4::setIdentity
        gas_idx_1 = i[1,1,3] - 1   # q1.u.w-1 
        ias_idx_1 = i[1,2,3] - 1   # q2.u.w-1 

        ins_idx_2 = i[2,0,3] - 1   # q0.u.w-1  see sqat4.h qat4::getIdentity qat4::setIdentity
        gas_idx_2 = i[2,1,3] - 1   # q1.u.w-1 
        ias_idx_2 = i[2,2,3] - 1   # q2.u.w-1 

        assert ins_idx_1 == ins_idx_2 
        assert gas_idx_1 == gas_idx_2 
        assert ias_idx_1 == ias_idx_2 

        ins_idx = ins_idx_1
        gas_idx = gas_idx_1
        ias_idx = ias_idx_1
        qat4id = "ins_idx %(ins_idx)6d gas_idx %(gas_idx)4d %(ias_idx)4d " % locals()

        self.ins_idx = ins_idx
        self.gas_idx = gas_idx
        self.ias_idx = ias_idx
        self.qat4id = qat4id

        m2w = a[1].copy()
        w2m = a[2].copy()

        if clear_identity:
            m2w[0,3] = 0. 
            m2w[1,3] = 0. 
            m2w[2,3] = 0. 
            m2w[3,3] = 1. 

            w2m[0,3] = 0. 
            w2m[1,3] = 0. 
            w2m[2,3] = 0. 
            w2m[3,3] = 1. 
        pass

        self.m2w = m2w
        self.w2m = w2m
        self.id = np.dot( m2w, w2m )  

    def __repr__(self):

        l_ = lambda k,v:"%-10s : %s" % (k, v) 

        return "\n".join(
                  [ l_("sframe",""),
                    l_("path",self.path), 
                    l_("meta",repr(self.meta)), 
                    l_("ce", repr(self.ce)), 
                    l_("grid", self.grid), 
                    l_("target", self.target), 
                    l_("qat4id", self.qat4id), 
                    l_("m2w",""), repr(self.m2w), "", 
                    l_("w2m",""), repr(self.w2m), "", 
                    l_("id",""),  repr(self.id) ])
    
 

if __name__ == '__main__':

    fr = sframe.Load()
    print(fr)

     

