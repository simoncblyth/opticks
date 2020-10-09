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
import os, logging
log = logging.getLogger(__name__)
import numpy as np
from opticks.ana.blib import BLib
from opticks.ana.key import key_

tx_load = lambda _:list(map(str.strip, open(_).readlines()))

def Three2Four(a, w=1): 
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


class GGeo(object):
    KEY = key_(os.environ["OPTICKS_KEY"])
    KEYDIR = KEY.keydir
    VERSION = KEY.version

    volume_names    = list(map(lambda _:"volume_%s" % _, "transforms center_extent bbox meshes nodeinfo identity".split()))
    placement_names = list(map(lambda _:"placement_%s"%_, "itransforms iidentity".split()))
    face_names      = list(map(lambda _:"face_%s"%_,  "sensors boundaries nodes indices".split()))
    vertex_names    = list(map(lambda _:"vertex_%s"%_,  "colors normals vertices".split()))
    names = volume_names + placement_names + face_names + vertex_names

    all_volume_names = list(map(lambda _:"all_volume_%s" % _, "nodeinfo identity center_extent bbox transforms".split()))

    @classmethod   
    def Path(cls, ridx, name, subdir="GMergedMesh", alldir="GNodeLib"): 
        keydir = cls.KEYDIR 
        if ridx == -1:
            fmt = "{keydir}/{alldir}/{name}.npy"
        elif ridx > -1: 
            fmt = "{keydir}/{subdir}/{ridx}/{name}.npy"
        else:
            assert 0
        pass 
        return os.path.expandvars(fmt.format(**locals()))

    @classmethod   
    def TxtPath(cls, name): 
        keydir = cls.KEYDIR 
        return os.path.expandvars("{keydir}/{name}".format(**locals()))

    @classmethod   
    def Array(cls, ridx, name): 
        path = cls.Path(ridx, name) 
        a = np.load(path)
        return a 

    @classmethod   
    def Txt(cls, name): 
        path = cls.TxtPath(name) 
        return np.array(tx_load(path)) 

    @classmethod   
    def Attn(cls, ridx, name): 
        return "_%s_%d" % (name, ridx) 

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

    def summary0(self):
        log.info("num_repeats:{gg.num_repeats}".format(gg=self))
        for ridx in range(self.num_repeats):
            for name in self.names:
                a = self.get_array(ridx,name)
                print("{ridx:2d} {name:25s}  {shape!r}".format(ridx=ridx,name=name,shape=a.shape))
            pass
            print()
        pass 

    def get_tot_volumes(self):
        return self.get_num_volumes(-1)

    def get_num_volumes(self, ridx):
        name = "all_volume_transforms" if ridx == -1 else "volume_transforms"
        a = self.get_array(ridx,name)
        return a.shape[0]

    def get_num_placements(self, ridx):
        if ridx == -1:
            return 1
        pass
        name = "placement_itransforms"
        a = self.get_array(ridx,name)
        return a.shape[0]

    def summary(self):
        """
        Array shapes
        """
        log.info("num_repeats:{gg.num_repeats}".format(gg=self))

        fmt = "{ridx:10d} {num_volumes:15d} {num_placements:15d} {num_placed_vol:15d}"
        print("%10s %15s %15s %15s" % ("ridx","num_volumes", "num_placements", "num_placed_vol"))
        tot_vol = 0 
        tot_vol_ = self.get_num_volumes(-1)
        for ridx in range(self.num_repeats):
            num_volumes = self.get_num_volumes(ridx)
            num_placements = self.get_num_placements(ridx)
            num_placed_vol = num_volumes*num_placements
            tot_vol += num_placed_vol
            print(fmt.format(**locals()))
        pass
        print("%10s %15s %15s %15d" % ("","","tot_vol:",tot_vol)) 
        print("%10s %15s %15s %15d" % ("","","tot_vol_:",tot_vol_)) 
        print()

        for name in self.names:
            shape = []
            for ridx in range(self.num_repeats):
                a = self.get_array(ridx,name)
                shape.append("{shape!r:20s}".format(shape=a.shape))
            pass     
            print("{name:25s} {shape}".format(name=name,shape="".join(shape)))
        pass
        for name in self.all_volume_names:
            ridx = -1
            a = self.get_array(ridx,name)
            print("{name:25s} {shape!r}".format(name=name,shape=a.shape))
        pass

    def get_all_transforms(self):
        """
        Access transforms of all volumes via triplet indexing.
        The ordering of the transforms is not the same as all_volume_transforms.
        However can still check the match using the identity info to find
        the node index.
        """
        tot_volumes = self.get_tot_volumes()
        tr = np.zeros([tot_volumes,4,4],dtype=np.float32)
        count = 0 
        for ridx in range(self.num_repeats):
            num_placements = self.get_num_placements(ridx)
            num_volumes = self.get_num_volumes(ridx)
            for pidx in range(num_placements):
                for oidx in range(num_volumes):
                    tr[count] = self.get_transform(ridx,pidx,oidx)
                    count += 1 
                pass
            pass
        pass
        assert tot_volumes == count  
        return tr  

    def get_transform(self, ridx, pidx, oidx):
        """
        :param ridx: repeat idx, 0 for remainder
        :param pidx: placement index of the instance, 0 for remainder
        :param oidx: offset index, within the instance or among the remainder

        DONE in get_all_transforms, verified both routes match for all nodes including the remainders
        """
        ## native (triplet access)
        placement_itransforms = self.get_array(ridx, "placement_itransforms") # identity for remainder
        volume_transforms = self.get_array(ridx, "volume_transforms")   # within the instance or from root for remainder
        itr = placement_itransforms[pidx]
        vtr = volume_transforms[oidx].reshape(4,4)

        ggt = np.dot( vtr, itr )  

        ## cross reference to the node index
        nidx = self.get_node_index(ridx,pidx,oidx)

        ## nodeindex access 
        all_volume_transforms = self.get_array(-1,"all_volume_transforms")
        ntr = all_volume_transforms[nidx]

        assert np.allclose( ggt, ntr )
        return ggt

    def get_node_index(self, ridx, pidx, oidx):
        """
        :param ridx: repeat index, 0 for remainder
        :param pidx: placement index of the instance, 0 for remainder
        :param oidx: offset index, within the instance or among the remainder
        :return nidx: all_volume node index 
        """
        placement_iidentity = self.get_array(ridx, "placement_iidentity")  # eg shape  (672, 5, 4)
        iid = placement_iidentity[pidx, oidx]
        nidx = iid[0]    
        return nidx 

    def get_triplet_index(self, nidx):
        """
        :param nidx: all_volume node index 
        :return ridx,pidx,oidx:

        Need to lay down this info, no easy way to go from node to triplet
        """
        return None


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
    mlibnames = property(lambda self:self.mlib[self.all_volume_identity[:,1]])   # mesh/lv names
    blibnames = property(lambda self:self.blib[self.all_volume_identity[:,2]])   # boundary names

    def __call__(self,*args):
        """
        A single integer argument is interpreted as a node index (nidx), 
        otherwise 2 or 3 args are interpreted as ridx,pidx,oidx with 
        oidx defaulting to zero if not provided.::

            gg(0,0,1000)   # triplet addressing to remainder volumes, NB all are ridx:0 pidx:0 
            gg(2792)       # same volume via node indexing 

            gg(5,0,2)      # first placement of cathode volume (DYB)
            gg(3201)       # same volume via node indexing   

            gg(5,671,2)    # last placement of cathode volume (DYB)
            gg(11410)      # same volume via node indexing 

        """
        if len(args) == 1: 
            nidx = args[0]
        elif len(args) == 2:
            ridx,pidx = args
            oidx = 0 
            nidx = self.get_node_index(ridx,pidx,oidx)
        elif len(args) == 3:
            ridx,pidx,oidx = args
            nidx = self.get_node_index(ridx,pidx,oidx)
        else:
            assert 0, "expecting argument of 1/2/3 integers"
        pass
        self.dump_node(nidx) 

    def dump_node(self,i):
        """
        :param i: all_volume node index 
        """
        gg = self

        iden = gg.all_volume_identity[i] 
        print("gg.identity[%d]  nidx/midx/bidx/sidx  %s  " % (i,iden) )
        nidx,midx,bidx,sidx = iden  
        print("gg.mlibnames[%d] : %s " % (i, gg.mlibnames[i]) )
        print("gg.blibnames[%d] : %s " % (i, gg.blibnames[i]) )

        bb = gg.all_volume_bbox[i]
        ce = gg.all_volume_center_extent[i]
        c4 = ce.copy()
        c4[3] = 1.
        
        tr = gg.all_volume_transforms[i]
        it = np.linalg.inv(tr) 
        ibb = np.dot( bb, it )   ## apply inverse transform to the volumes bbox (mn,mx), should give symmetric (mn,mx)   
        cbb = (bb[0]+bb[1])/2.   ## center of bb should be same as c4
        assert np.allclose( c4, cbb )

        ic4 = np.dot( c4, it )   ## should be close to origin
        gt = gg.all_volume_transforms[nidx]  


        print("\ngt : gg.all_volume_transforms[%d]" % nidx)
        print(gt)
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
        print("\nic4 : np.dot( c4, it) : inverse transform applied to center4 : expect close to origin ")
        print(ic4)
        print("\nibb : np.dot( bb, it) : inverse transform applied to bbox4 : expect symmetric around origin")
        print(ibb)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(suppress=True, linewidth=200) 
    gg = GGeo()
    bbox = gg.all_volume_bbox
    print("gg.all_volume_bbox\n",bbox)

    t000 = gg.get_transform(0,0,0)
    print("t000\n",t000)

    #gg.summary0()
    gg.summary()
    tr = gg.get_all_transforms()



 
