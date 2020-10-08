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
from opticks.ana.key import key_

tx_load = lambda _:list(map(str.strip, open(_).readlines()))

class GGeo(object):
    KEY = key_(os.environ["OPTICKS_KEY"])
    KEYDIR = KEY.keydir
    VERSION = KEY.version

    key2aname = {
             "bbox":"volume_bbox",
            "bbox4":"volume_bbox",
          "center4":"volume_center_extent",
       "transforms":"volume_transforms",
      "itransforms":"placement_itransforms",
    }

    @classmethod   
    def Path(cls, ridx, name, reldir="GMergedMesh", alldir="GNodeLib"): 
        if name.endswith("raw"):
            name = name[:-3]
        pass 
        aname = cls.key2aname.get(name, name) 
        keydir = cls.KEYDIR 
        if ridx == -1:
            fmt = "{keydir}/{alldir}/{aname}.npy"
        elif ridx > -1: 
            fmt = "{keydir}/{reldir}/{ridx}/{aname}.npy"
        else:
            assert 0
        pass 
        return os.path.expandvars(fmt.format(**locals()))

    @classmethod   
    def TxtPath(cls, name): 
        keydir = cls.KEYDIR 
        return os.path.expandvars("{keydir}/{name}".format(**locals()))

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
        #r = cls.Reshape(a, name)
        #if name.endswith("4"):
        #   r = cls.Three2Four(r)
        #pass
        return a 

    @classmethod   
    def Txt(cls, name): 
        path = cls.TxtPath(name) 
        return np.array(tx_load(path)) 

    @classmethod   
    def Attn(cls, ridx, name): 
        attn = "_%s_%d" % (name, ridx) 
        return attn  

    def __init__(self):
        keydir = self.KEYDIR
        path = os.path.expandvars("{keydir}/GMergedMesh".format(**locals()))
        mmidx = sorted(map(int,os.listdir(path)))
        num_repeats = len(mmidx)
        self.num_repeats = num_repeats 
        blib = BLib(keydir)
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


    def lookup(self,ridx,pidx,oidx):
        """
        :param ridx: repeat idx or 0 for global remainder
        :param pidx: placement idx, will be 0 for all globals
        :param oidx: offset index within the instance or amoung the global volumes 
        """
        gc = self
        ggt = gc.get_transform(ridx,pidx,oidx)
        print("\nggt : gc.get_transform(%d,%d,%d)" % (ridx,pidx,oidx))
        print(ggt)

    def __call__(self,i):
        """
        Focussing on accessing globals 
        """
        gg = self

        iden = gg.identity[i] 
        print("gg.identity[%d]  nidx/midx/bidx/sidx  %s  " % (i,iden) )
        nidx,midx,bidx,sidx = iden  
        print("gg.mlibnames[%d] : %s " % (i, gg.mlibnames[i]) )
        print("gg.blibnames[%d] : %s " % (i, gg.blibnames[i]) )

        bb = gg.bbox[i]
        ce = gg.center_extent[i]
        c4 = gg.center4[i]
        tr = gg.transforms[i]
        it = np.linalg.inv(tr) 
        ibb = np.dot( bb, it )  
        cbb = (bb[0]+bb[1])/2.

        ic4 = np.dot( c4, it )

        gt = gg.transforms0[nidx]  

        ggt = gg.get_transform(0,0,i)


        print("\ngt : gg.transforms0[%d]" % nidx)
        print(gt)

        print("\nggt : gg.get_transform(0,0,%d)" % i)
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


    all_volume_center_extent = property(lambda self:self.get_array(-1,"all_volume_center_extent"))
    all_volume_bbox          = property(lambda self:self.get_array(-1,"all_volume_bbox"))
    all_volume_identity      = property(lambda self:self.get_array(-1,"all_volume_identity"))
    all_volume_transforms    = property(lambda self:self.get_array(-1,"all_volume_transforms"))  

    volume_transforms0  = property(lambda self:self.get_array(0,"volume_transforms")) 
    volume_transforms1  = property(lambda self:self.get_array(1,"volume_transforms"))
    volume_transforms2  = property(lambda self:self.get_array(2,"volume_transforms"))
    volume_transforms3  = property(lambda self:self.get_array(3,"volume_transforms"))
    volume_transforms4  = property(lambda self:self.get_array(4,"volume_transforms"))
    volume_transforms5  = property(lambda self:self.get_array(5,"volume_transforms"))

    placement_itransforms0 = property(lambda self:self.get_array(0,"placement_itransforms"))
    placement_itransforms1 = property(lambda self:self.get_array(1,"placement_itransforms"))
    placement_itransforms2 = property(lambda self:self.get_array(2,"placement_itransforms"))
    placement_itransforms3 = property(lambda self:self.get_array(3,"placement_itransforms"))
    placement_itransforms4 = property(lambda self:self.get_array(4,"placement_itransforms"))
    placement_itransforms5 = property(lambda self:self.get_array(5,"placement_itransforms"))

    placement_iidentity0 = property(lambda self:self.get_array(0,"placement_iidentity"))
    placement_iidentity1 = property(lambda self:self.get_array(1,"placement_iidentity"))
    placement_iidentity2 = property(lambda self:self.get_array(2,"placement_iidentity"))
    placement_iidentity3 = property(lambda self:self.get_array(3,"placement_iidentity"))
    placement_iidentity4 = property(lambda self:self.get_array(4,"placement_iidentity"))
    placement_iidentity5 = property(lambda self:self.get_array(5,"placement_iidentity"))

    mlib = property(lambda self:self.get_txt("GItemList/GMeshLib.txt", "_mlib")) 
    mlibnames = property(lambda self:self.mlib[self.identity[:,1]])
    blibnames = property(lambda self:self.blib[self.identity[:,2]])

    def get_transform(self, ridx, pidx, oidx):
        """
        :param ridx: repeat index
        :param pidx: placement index of the instance
        :param oidx: offset index, within the instance

        TODO: verify both routes match for all nodes including the remainders
        """
        ## native (triplet access)
        placement_itransforms = self.get_array(ridx, "placement_itransforms")
        volume_transforms = self.get_array(ridx, "volume_transforms") 
        itr = placement_itransforms[pidx]
        vtr = volume_transforms[oidx].reshape(4,4)

        print("vtr\n",vtr.shape)

        ggt = np.dot( vtr, itr )  

        ## cross reference to the node index
        placement_iidentity = self.get_array(ridx, "placement_iidentity")  # eg shape  (672, 5, 4)
        iid = placement_iidentity[pidx, oidx]
        nidx = iid[0]    

        ## nodeindex access 
        all_volume_transforms = self.get_array(-1,"all_volume_transforms")
        ntr = all_volume_transforms[nidx]

        assert np.allclose( ggt, ntr )
        return ggt


if __name__ == '__main__':
    np.set_printoptions(suppress=True) 
    gg = GGeo()
    bbox = gg.all_volume_bbox
    print("gg.all_volume_bbox\n",bbox)

    t100 = gg.get_transform(1,0,0)
    print("t100\n",t100)

    t000 = gg.get_transform(0,0,0)
    print("t000\n",t000)
 
 
