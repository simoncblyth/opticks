#!/usr/bin/env python
"""
ggeo.py
=========

See also GNodeLib.py 

Dumping using single node index and triplet RPO (ridx/pidx/oidx repeat/placement/offset) indexing::

     ggeo.py 0         # world volume 
     ggeo.py 1/0/0     # first placement of outer volume of first repeat   
     ggeo.py 1/        # missing elements of the triplet default to 0 

     ggeo.py 2/0/0     # first placement of outer volume of second repeat   

A convenient visualization workflow is to use the above python triple indexing to find 
flat node indices to target and then use, eg::

    OTracerTest --target 69078


Volume idsmry dumping::

    epsilon:ana blyth$ ggeo.py 3199 -i
    iden(  3199    5000000     2f001b     -1 )  nrpo(  3199     5     0     0 )  shape(  47  27                pmt-hemi0xc0fed900x3e85f00                       MineralOil///Pyrex) 
    iden(  3200    5000001     2e001c     -1 )  nrpo(  3200     5     0     1 )  shape(  46  28            pmt-hemi-vac0xc21e2480x3e85290                           Pyrex///Vacuum) 
    iden(  3201    5000002     2b001d     -1 )  nrpo(  3201     5     0     2 )  shape(  43  29        pmt-hemi-cathode0xc2f1ce80x3e842d0                        Vacuum///Bialkali) 
    iden(  3202    5000003     2c001e     -1 )  nrpo(  3202     5     0     3 )  shape(  44  30            pmt-hemi-bot0xc22a9580x3e844c0                    Vacuum///OpaqueVacuum) 
    iden(  3203    5000004     2d001e     -1 )  nrpo(  3203     5     0     4 )  shape(  45  30         pmt-hemi-dynode0xc346c500x3e84610                    Vacuum///OpaqueVacuum) 
    iden(  3204        57f     30001f     -1 )  nrpo(  3204     0     0  1407 )  shape(  48  31             AdPmtCollar0xc2c52600x3e86030          MineralOil///UnstStainlessSteel) 
    iden(  3205    5000100     2f001b     -1 )  nrpo(  3205     5     1     0 )  shape(  47  27                pmt-hemi0xc0fed900x3e85f00                       MineralOil///Pyrex) 
    iden(  3206    5000101     2e001c     -1 )  nrpo(  3206     5     1     1 )  shape(  46  28            pmt-hemi-vac0xc21e2480x3e85290                           Pyrex///Vacuum) 
    iden(  3207    5000102     2b001d     -1 )  nrpo(  3207     5     1     2 )  shape(  43  29        pmt-hemi-cathode0xc2f1ce80x3e842d0                        Vacuum///Bialkali) 


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
import os, logging, argparse
log = logging.getLogger(__name__)
import numpy as np
from opticks.ana.blib import BLib
from opticks.ana.key import key_
from opticks.ana.OpticksIdentity import OpticksIdentity

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

    all_volume_names = list(map(lambda _:"all_volume_%s" % _, "nodeinfo identity center_extent bbox transforms inverse_transforms".split()))

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

    def __init__(self, args=None):
        self.args = args
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
    num_volumes = property(get_tot_volumes)

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
        log.info("get_all_transforms and do identity consistency check : triplet->node->triplet")
        tot_volumes = self.get_tot_volumes()
        tr = np.zeros([tot_volumes,4,4],dtype=np.float32)
        count = 0 
        for ridx in range(self.num_repeats):
            num_placements = self.get_num_placements(ridx)
            num_volumes = self.get_num_volumes(ridx)
            for pidx in range(num_placements):
                for oidx in range(num_volumes):
                    nidx = self.get_node_index(ridx,pidx,oidx)
                    tr[nidx] = self.get_transform(ridx,pidx,oidx)
                    count += 1 
                pass
            pass
        pass
        assert tot_volumes == count  
        all_volume_transforms = self.get_array(-1,"all_volume_transforms")
        assert np.allclose( all_volume_transforms, tr )
        return tr  


    def get_transform_n(self, nidx):
        all_volume_transforms = self.get_array(-1,"all_volume_transforms")
        return all_volume_transforms[nidx] 

    def get_inverse_transform_n(self, nidx):
        all_volume_inverse_transforms = self.get_array(-1,"all_volume_inverse_transforms")
        return all_volume_inverse_transforms[nidx] 


    def get_inverse_transform(self, ridx, pidx, oidx):
        """
        No triplet way to do this yet, have to go via node index
        """
        nidx = self.get_node_index(ridx,pidx,oidx)
        return self.get_inverse_transform_n(nidx)

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
        ntr = self.get_transform_n(nidx)

        assert np.allclose( ggt, ntr )
        return ggt

    def get_node_index(self, ridx, pidx, oidx):
        """
        :param ridx: repeat index, 0 for remainder
        :param pidx: placement index of the instance, 0 for remainder
        :param oidx: offset index, within the instance or among the remainder
        :return nidx: all_volume node index 

        The node index obtained from the placement_identity is used
        to do a reverse conversion check using nrpo, looking up 
        the triplet identity from the node index. These indices
        are consistency checked with the inputs.
        """
        placement_iidentity = self.get_array(ridx, "placement_iidentity")  # eg shape  (672, 5, 4)
        iid = placement_iidentity[pidx, oidx]
        nidx = iid[0]    

        nidx2,ridx2,pidx2,oidx2 = self.nrpo[nidx]
        assert nidx2 == nidx 
        assert ridx2 == ridx 
        assert pidx2 == pidx 
        assert oidx2 == oidx 

        return nidx 

    def make_nrpo(self):
        """
        See okc/OpticksIdentity::Decode
        """
        gg = self
        avi = gg.all_volume_identity
        tid = avi[:,1] 
        nrpo = OpticksIdentity.NRPO(tid)
        return nrpo

    def _get_nrpo(self):
        if getattr(self,'_nrpo',None) is None:
            setattr(self,'_nrpo',self.make_nrpo())
        return self._nrpo 
    nrpo = property(_get_nrpo)

    def get_triplet_index(self, nidx):
        """
         cf ggeo/GGeo::getIdentity

        :param nidx: all_volume node index 
        :return nidx,ridx,pidx,oidx:
        """
        return self.nrpo[nidx]


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

    midx = property(lambda self:(self.all_volume_identity[:,2] >> 16) & 0xffff ) 
    bidx = property(lambda self:(self.all_volume_identity[:,2] >>  0) & 0xffff ) 

    mlibnames = property(lambda self:self.mlib[self.midx])   # mesh/lv names
    blibnames = property(lambda self:self.blib[self.bidx])   # boundary names


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

        if self.args.nidx:
            print(nidx)
        else: 
            self.dump_node(nidx) 
        pass


    def idsmry(self, nidx):
        """
        volume identity summary 
        """
        iden = self.all_volume_identity[nidx]
        nidx2,triplet,shape,_ = iden  
        assert nidx == nidx2
        sidx = iden[-1].view(np.int32)  
        iden_s = "nidx:{nidx} triplet:{triplet:7x} sh:{sh:x} sidx:{sidx:5d} ".format(nidx=nidx,triplet=triplet,sh=shape,sidx=sidx)

        nrpo = self.nrpo[nidx]
        nrpo_s = "nrpo( %5d %5d %5d %5d )" % tuple(nrpo)

        midx = self.midx[nidx] 
        bidx = self.bidx[nidx] 
        shape_s = "shape( %3d %3d  %40s %40s)" % (midx,bidx, self.mlibnames[nidx], self.blibnames[nidx] )
        print( "%s  %s  %s " % (iden_s, nrpo_s, shape_s) )


    def bbsmry(self, nidx):
        gg = self
        bb = gg.all_volume_bbox[nidx]
        ce = gg.all_volume_center_extent[nidx]
        print(" {nidx:5d} {ce!s:20s} ".format(nidx=nidx,bb=bb,ce=ce))


    def dump_node(self,nidx):
        gg = self
        gg.idsmry(nidx)

        #print("gg.mlibnames[%d] : %s " % (i, gg.mlibnames[nidx]) )
        #print("gg.blibnames[%d] : %s " % (i, gg.blibnames[nidx]) )

        bb = gg.all_volume_bbox[nidx]
        ce = gg.all_volume_center_extent[nidx]
        c4 = ce.copy()
        c4[3] = 1.
        
        tr = gg.all_volume_transforms[nidx]
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

    def consistency_check(self):
        gg = self
        log.info("consistency_check")
        gg.summary()

        tr = gg.get_all_transforms()



def parse_args(doc, **kwa):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    parser = argparse.ArgumentParser(doc)
    parser.add_argument(     "idx", nargs="*", help="Node index or triplet index of form \"1/0/0\" or \"1/\" to dump.")
    parser.add_argument(     "--nidx", default=False, action="store_true", help="Dump only the node index, useful when the input is triplet index." ) 
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    parser.add_argument(  "-c","--check", action="store_true", help="Consistency check" ) 
    parser.add_argument(  "-i","--idsmry", action="store_true", help="Slice identity summary interpreting idx as slice range." ) 
    parser.add_argument(  "-b","--bbsmry", action="store_true", help="Slice bbox summary interpreting idx as slice range." ) 
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    if len(args.idx) == 0:
        args.idx = [0]
    pass
    return args  


def misc(gg):
    bbox = gg.all_volume_bbox
    print("gg.all_volume_bbox\n",bbox)
    t000 = gg.get_transform(0,0,0)
    print("t000\n",t000)

def triplet_(rpo):
    elem = []
    for s in rpo.split("/"):
        try:
            elem.append(int(s))
        except ValueError:
            elem.append(0)
        pass
    pass
    return elem 


if __name__ == '__main__':
    args = parse_args(__doc__)
    gg = GGeo(args)

    if args.check:
        gg.consistency_check()
    elif args.idsmry or args.bbsmry:

        beg = int(args.idx[0])
        end = int(args.idx[1]) if len(args.idx) > 1 else min(gg.num_volumes,int(args.idx[0])+50) 
        for idx in list(range(beg, end)):
            if args.idsmry:
                gg.idsmry(idx)
            elif args.bbsmry:
                gg.bbsmry(idx)
            else:
                pass
            pass
        pass
    else:
        for idx in args.idx:
            try:
                gg(int(idx))
            except ValueError:
                gg(*triplet_(idx))
            pass  
        pass
    pass

 
