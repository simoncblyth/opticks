#!/usr/bin/env python
"""
::

    In [52]: gc(10)
    gc.identity[10]  nidx/midx/bidx/sidx  [530   8   3   1]  
    gc.mlibnames[10] : sPlane0x47c46c0 
    gc.blibnames[10] : Air///Air 

    gt : gc.transforms0[530]
    [[    0.       1.       0.       0.  ]
     [   -1.       0.       0.       0.  ]
     [    0.       0.       1.       0.  ]
     [20133.6  -6711.2  23504.15     1.  ]]

    tr : transform
    [[    0.       1.       0.       0.  ]
     [   -1.       0.       0.       0.  ]
     [    0.       0.       1.       0.  ]
     [20133.6  -6711.2  23504.15     1.  ]]

    it : inverted transform
    [[     0.       -1.        0.        0.  ]
     [     1.        0.        0.        0.  ]
     [    -0.        0.        1.        0.  ]
     [  6711.2   20133.6  -23504.15      1.  ]]

    bb : bbox4
    [[ 16748.4492 -10141.8008  23497.5         1.    ]
     [ 23518.75    -3280.6001  23510.8008      1.    ]]

    cbb : (bb[0]+bb[1])/2.
    [20133.5996 -6711.2004 23504.1504     1.    ]

    c4 : center4
    [20133.5996 -6711.2002 23504.1504     1.    ]

    ce : center_extent
    [20133.6    -6711.2    23504.15    3430.6003]

    ic4 : np.dot( c4, it) : inverse transform applied to center4 
    [0. 0. 0. 1.]

    ibb : np.dot( bb, it) : inverse transform applied to bbox4 
    [[-3430.6006  3385.1504    -6.6504     1.    ]
     [ 3430.6001 -3385.1504     6.6504     1.    ]]


"""
import os
import numpy as np
from opticks.ana.blib import BLib

tx_load = lambda _:list(map(str.strip, open(_).readlines()))

class GGeo(object):

    key2aname = {
      "bbox4":"bbox",
      "center4":"center_extent",
    }

    @classmethod   
    def Path(cls, ridx, name): 
        if name.endswith("raw"):
            name = name[:-3]
        pass 
        aname = cls.key2aname.get(name, name) 
        return os.path.expandvars("$GC/GMergedMesh/%d/%s.npy" % (ridx, aname))

    @classmethod   
    def TxtPath(cls, name): 
        return os.path.expandvars("$GC/%s" % (name))

    @classmethod   
    def Three2Four(cls, a, w=1): 
        """
        :param a: array shaped with last dimension 3 
        :param w: 1 or 0 (points or vectors)
        :return r: corresponding array which last dimension increased from 3 to 4
        """
        s = list(a.shape)
        assert s[-1] == 3, "unexpected shape %r , last dimension must be 3" % s
        assert w in (1,0), "w must be 1 or 0" 
        s[-1] = 4 
        b = np.ones(s) if w == 1 else np.zeros(s)
        d = len(s)
        if d == 1:
            b[:3] = a
        if d == 2:
            b[:,:3] = a
        elif d == 3:
            b[:,:,:3] = a
        elif d == 4:
            b[:,:,:,:3] = a
        else:
            assert 0, "unexpected shape %r " % s
        pass
        r = b
        return r 

    @classmethod   
    def Reshape(cls, a, name): 
        if name.endswith("raw"):  # no reshaping 
            r = a  
        elif name == "bbox":
            r = a.reshape(-1,2,3)
        elif name == "bbox4":
            r = a.reshape(-1,2,3)
        elif name == "transforms":
            r = a.reshape(-1,4,4)
        elif name == "identity":
            r = a
        elif name == "center4" or name == "center":
            r = a[:,:3].copy()
        else:
            r = a 
        pass
        return r 


    @classmethod   
    def Array(cls, ridx, name): 
        path = cls.Path(ridx, name) 
        a = np.load(path)
        r = cls.Reshape(a, name)
        if name.endswith("4"):
           r = cls.Three2Four(r)
        pass
        return r 

    @classmethod   
    def Txt(cls, name): 
        path = cls.TxtPath(name) 
        return np.array(tx_load(path)) 

    @classmethod   
    def Attn(cls, ridx, name): 
        attn = "_%s_%d" % (name, ridx) 
        return attn  

    def __init__(self):
        mmidx = sorted(map(int,os.listdir(os.path.expandvars("$GC/GMergedMesh"))))
        mmmx = mmidx[-1]
        self.mmmx = mmmx 
        blib = BLib("$GC")
        self.blib = np.array(blib.names().split("\n"))

    def get_array(self, ridx, name):
        """
        Array actually loaded only the first time
        """
        attn = self.Attn(ridx,name) 
        if getattr(self, attn, None) is None:
            a = self.Array(ridx, name)
            setattr(self, attn, a)
        pass
        return getattr(self, attn)

    def get_txt(self, name, attn):
        if getattr(self, attn, None) is None:
            a = self.Txt(name)
            setattr(self, attn, a)
        pass
        return getattr(self, attn)


    def lookup(self,ridx,iidx,vidx):
        """
        :param ridx: repeat idx or 0 for global remainder
        :param iidx: instance idx, will be 0 for all globals
        :param vidx: volume index within the instance or amoung the global volumes 
        """
        gc = self
        ggt = gc.get_transform(ridx,iidx,vidx)
        print("\nggt : gc.get_transform(%d,%d,%d)" % (ridx,iidx,vidx))
        print(ggt)

          





    def __call__(self,i):
        """
        Focussing on accessing globals 
        """
        gc = self

        iden = gc.identity[i] 
        print("gc.identity[%d]  nidx/midx/bidx/sidx  %s  " % (i,iden) )
        nidx,midx,bidx,sidx = iden  
        print("gc.mlibnames[%d] : %s " % (i, gc.mlibnames[i]) )
        print("gc.blibnames[%d] : %s " % (i, gc.blibnames[i]) )

        bb = gc.bbox4[i]
        ce = gc.center_extent[i]
        c4 = gc.center4[i]
        tr = gc.transforms[i]
        it = np.linalg.inv(tr) 
        ibb = np.dot( bb, it )  
        cbb = (bb[0]+bb[1])/2.

        ic4 = np.dot( c4, it )

        gt = gc.transforms0[nidx]  

        ggt = gc.get_transform(0,0,i)


        print("\ngt : gc.transforms0[%d]" % nidx)
        print(gt)

        print("\nggt : gc.get_transform(0,0,%d)" % i)
        print(ggt)

        print("\ntr : transform")
        print(tr)
        print("\nit : inverted transform")
        print(it)
        print("\nbb : bbox4")
        print(bb)
        print("\ncbb : (bb[0]+bb[1])/2.")
        print(cbb)
        print("\nc4 : center4")
        print(c4)
        print("\nce : center_extent")
        print(ce)
        print("\nic4 : np.dot( c4, it) : inverse transform applied to center4 ")
        print(ic4)
        print("\nibb : np.dot( bb, it) : inverse transform applied to bbox4 ")
        print(ibb)

 
    center_extent = property(lambda self:self.get_array(self.mmmx,"center_extent"))
    center4     = property(lambda self:self.get_array(self.mmmx,"center4"))
    bboxraw     = property(lambda self:self.get_array(self.mmmx,"bboxraw"))
    bbox        = property(lambda self:self.get_array(self.mmmx,"bbox"))
    bbox4       = property(lambda self:self.get_array(self.mmmx,"bbox4"))
    identity    = property(lambda self:self.get_array(self.mmmx,"identity"))

    transforms0  = property(lambda self:self.get_array(0,"transforms"))  # all node globals
    transforms1  = property(lambda self:self.get_array(1,"transforms"))
    transforms2  = property(lambda self:self.get_array(2,"transforms"))
    transforms3  = property(lambda self:self.get_array(3,"transforms"))
    transforms4  = property(lambda self:self.get_array(4,"transforms"))
    transforms5  = property(lambda self:self.get_array(5,"transforms"))
    transforms   = property(lambda self:self.get_array(self.mmmx,"transforms"))  # ridx 0 just globals 

    itransforms0 = property(lambda self:self.get_array(0,"itransforms"))
    itransforms1 = property(lambda self:self.get_array(1,"itransforms"))
    itransforms2 = property(lambda self:self.get_array(2,"itransforms"))
    itransforms3 = property(lambda self:self.get_array(3,"itransforms"))
    itransforms4 = property(lambda self:self.get_array(4,"itransforms"))
    itransforms5 = property(lambda self:self.get_array(5,"itransforms"))

    iidentity0 = property(lambda self:self.get_array(0,"iidentity"))
    iidentity1 = property(lambda self:self.get_array(1,"iidentity"))
    iidentity2 = property(lambda self:self.get_array(2,"iidentity"))
    iidentity3 = property(lambda self:self.get_array(3,"iidentity"))
    iidentity4 = property(lambda self:self.get_array(4,"iidentity"))
    iidentity5 = property(lambda self:self.get_array(5,"iidentity"))

    mlib = property(lambda self:self.get_txt("GItemList/GMeshLib.txt", "_mlib")) 
    mlibnames = property(lambda self:self.mlib[self.identity[:,1]])
    blibnames = property(lambda self:self.blib[self.identity[:,2]])

    def get_transform(self, ridx, iidx, vidx):
        """
        """
        iidentity = self.get_array(ridx, "iidentity")
        nidx0 = iidentity[iidx,vidx,0]
        iid = iidentity[iidx, vidx]
        nidx = iid[0] 
        assert nidx == nidx0, (nidx, nidx0)

        transforms0 = self.transforms0
        ntr = transforms0[nidx]

        itransforms = self.get_array(ridx, "itransforms")

        mmmx = self.mmmx
        transforms = self.get_array(ridx if ridx > 0 else mmmx, "transforms") 

        itr = itransforms[iidx]
        vtr = transforms[vidx]
        ggt = np.dot( vtr, itr )  

        assert np.allclose( ggt, ntr )
        return ggt


if __name__ == '__main__':
    gc = GGeo()
    bbox = gc.bbox
    print(bbox)


 
