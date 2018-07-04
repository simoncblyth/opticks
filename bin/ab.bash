ab-source(){ echo $BASH_SOURCE ; }
ab-vi(){ vi $(ab-source)  ; }
ab-env(){  olocal- ; opticks- ; }
ab-usage(){ cat << EOU

ab Usage 
===================


EOU
}


ab-base(){  echo  /usr/local/opticks/geocache ; }
ab-cd(){   cd $(ab-base) ; }

ab-tmp(){ echo /tmp/$USER/opticks/bin/ab ; }

ab-a-dir(){ echo DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd  ; }
ab-b-dir(){ echo OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe ; }
ab-a-idpath(){ echo $(ab-base)/$(ab-a-dir)/103 ; }
ab-b-idpath(){ echo $(ab-base)/$(ab-b-dir)/1 ; }


#ab-tail(){ echo ${AB_TAIL:-.} ; }
ab-tail(){ echo ${AB_TAIL:-0} ; }

ab-a-(){ echo $(ab-a-idpath)/GPartsAnalytic/$(ab-tail) ; }
ab-b-(){ echo $(ab-b-idpath)/GParts/$(ab-tail) ; }

ab-a(){  cd $(ab-a-); }
ab-b(){  cd $(ab-b-); }

ab-diff()
{  
   ab-cd
   diff -r --brief $(ab-a-) $(ab-b-) 

   ab-a 
   np.py
   md5 *

   ab-b
   np.py
   md5 *
}

ab-blib()
{
   blib.py $(ab-a-idpath)
   blib.py $(ab-b-idpath)
}
ab-prim(){
   prim.py $(ab-a-)
   prim.py $(ab-b-)
}




ab-genrun(){
    local func=$1
    mkdir -p $(ab-tmp)
    local py=$(ab-tmp)/$func.py
    $func- $* > $py 
    cat $py
    ipython -i $py 
}
ab-i(){ ab-genrun $FUNCNAME ; }
ab-p(){ ab-genrun $FUNCNAME ; }
ab-t(){ ab-genrun $FUNCNAME ; }
ab-s(){ ab-genrun $FUNCNAME ; }


ab-p-(){ cat << EOP
import os, numpy as np
from opticks.ana.mesh import Mesh
from opticks.ana.prim import Dir
from opticks.sysrap.OpticksCSG import CSG_

a_dir = "$(ab-a-)"
b_dir = "$(ab-b-)"
a_idpath = "$(ab-a-idpath)"
b_idpath = "$(ab-b-idpath)"

a_load = lambda _:np.load(os.path.join(a_dir, _))
b_load = lambda _:np.load(os.path.join(b_dir, _))

pa = a_load("primBuffer.npy")
pb = b_load("primBuffer.npy")
assert np.all( pa == pb )

xb = b_load("idxBuffer.npy")
assert len(pa) == len(xb)

ma = Mesh.make(a_idpath)
nn = np.load(os.path.expandvars("$TMP/NNodeNudger.npy"))

print "nudged (lvIdx/treeidx,num_prim,coincidences,nudges)\n nn[np.where( nn[:,3] > 0 )] \n", nn[np.where( nn[:,3] > 0 )]

da = Dir(a_dir)
db = Dir(b_dir)

cut = 0.1
where_discrepant = da.where_discrepant_prims(db, cut) 
wd = np.array(where_discrepant, dtype=np.uint32)
lvd = np.unique(xb[wd][:,2])

print " num_discrepant %d cut %s " % ( len(where_discrepant), cut ) 

for i in where_discrepant:

    primIdx = i 
    _,soIdx,lvIdx,height = xb[i]
    name = ma.idx2name[lvIdx]

    print " %s primIdx:%3d soIdx:%3d lvIdx:%3d height:%d name:%s  %s " % ( "-" * 30, primIdx, soIdx,lvIdx,height, name,   "-" * 60 )
    dap = da.prims[i]
    dbp = db.prims[i]
    print dap.maxdiff(dbp)
    print dap
    print dbp
    print
    print

EOP
}


ab-s-(){ cat << EOP

import os, numpy as np

aa = np.load(os.path.expandvars("$TMP/Boolean_all_transforms.npy"))
bb = np.load(os.path.expandvars("$TMP/X4Transform3D.npy"))

assert np.all( aa[:,3] == bb[:,3] )  ## translation matches
assert aa.shape == bb.shape 

cut = 1e-19
discrep = 0 

"""
X4Transform3D::GetDisplacementTransform

* transposing C++ rotation gives agreement when using disp->GetObjectRotation()

::

     43 glm::mat4 X4Transform3D::GetDisplacementTransform(const G4DisplacedSolid* const disp)
     44 {
     45     if(TranBuffer == NULL) TranBuffer = NPY<float>::make(0,4,4) ;
     46 
     47     G4RotationMatrix rot = disp->GetFrameRotation();
     48     //G4RotationMatrix rot = disp->GetObjectRotation();    // started with this, see notes/issues/OKX4Test_tranBuffer_mm0.rst
     49     LOG(error) << "GetDisplacementTransform rot " << rot ;
     50 

"""
for _ in range(len(aa)):
    a = aa[_][:3,:3]
    b = bb[_][:3,:3]

    #ab = a - b.T   
    ab = a - b 

    mx = np.max( ab )
    mi = np.min( ab ) 

    if abs(mx) < cut and abs(mi) < cut:
        pass
    else:
        discrep +=  1
        print _, mx, mi
        print np.hstack([a,b])
    pass
pass

print " cut %s discrep %s " % ( cut, discrep ) 



EOP
}

ab-t-(){ cat << EOP
import os, numpy as np
from opticks.ana.mesh import Mesh
from opticks.ana.prim import Dir
from opticks.sysrap.OpticksCSG import CSG_

a_dir = "$(ab-a-)"
b_dir = "$(ab-b-)"


da = Dir(a_dir)
db = Dir(b_dir)

a_idpath = "$(ab-a-idpath)"
b_idpath = "$(ab-b-idpath)"

a_load = lambda _:np.load(os.path.join(a_dir, _))
b_load = lambda _:np.load(os.path.join(b_dir, _))

a = a_load("partBuffer.npy")
ta = a_load("tranBuffer.npy")
pa = a_load("primBuffer.npy")

b = b_load("partBuffer.npy")
tb = b_load("tranBuffer.npy")
pb = b_load("primBuffer.npy")

xb = b_load("idxBuffer.npy")

ma = Mesh.make(a_idpath)
mb = Mesh.make(b_idpath)


cut = 0.1
where_discrepant_tr = da.where_discrepant_tr(db, cut) 
wd = np.array(where_discrepant_tr, dtype=np.uint32)
lvd = np.unique(xb[wd][:,2])

print " prim with discrepant_tr %d cut %s " % ( len(where_discrepant_tr), cut ) 


detail = False

lvs = set()

for i in where_discrepant_tr:

    primIdx = i 
    _,soIdx,lvIdx,height = xb[i]
    name = ma.idx2name[lvIdx]

    print " %s primIdx:%3d soIdx:%3d lvIdx:%3d height:%d name:%s  %s " % ( "-" * 30, primIdx, soIdx,lvIdx,height, name,   "-" * 60 )
    lvs.add(lvIdx)
    if detail:
        dap = da.prims[i]
        dbp = db.prims[i]
        print dap.tr_maxdiff(dbp)
        print dap
        print dbp
        print
        print
    pass
pass


lvs = np.array( sorted(list(lvs)), dtype=np.uint32 )
print "lvs: %s " % repr(lvs)




EOP
}


ab-i-(){ cat << EOP

import os, logging, numpy as np
log = logging.getLogger(__name__)
from opticks.ana.mesh import Mesh
from opticks.ana.prim import Dir
from opticks.sysrap.OpticksCSG import CSG_

logging.basicConfig(level=logging.INFO)


a_dir = "$(ab-a-)"
b_dir = "$(ab-b-)"


a_load = lambda _:np.load(os.path.join(a_dir, _))
b_load = lambda _:np.load(os.path.join(b_dir, _))

a = a_load("partBuffer.npy")
ta = a_load("tranBuffer.npy")
pa = a_load("primBuffer.npy")
la = a_load("planBuffer.npy")

b = b_load("partBuffer.npy")
tb = b_load("tranBuffer.npy")
pb = b_load("primBuffer.npy")
lb = b_load("planBuffer.npy")


msg = "shape mismatch can be caused by runnig with an active OPTICKS_QUERY_LIVE selection "
assert a.shape == b.shape, msg 
assert ta.shape == tb.shape, msg 
assert pa.shape == pb.shape, msg  




da = Dir(a_dir)
db = Dir(b_dir)

a_idpath = "$(ab-a-idpath)"
b_idpath = "$(ab-b-idpath)"
xb = b_load("idxBuffer.npy")

ma = Mesh.make(a_idpath)
mb = Mesh.make(b_idpath)


def cfprim_debug(pa,pb,xb,ma):
    w = np.where( pa[:,1] != pb[:,1] )[0]

    lv = np.unique(xb[w][:,2])

    print "\n".join(map(lambda _:ma.idx2name[_], lv ))



def cfprim(pa,pb,xb,ma):
    """
    high level comparison of prim buffers

    primBuffer will be matched when all prim trees have same heights
    and the usage of tranforms and planes within each prim are the same

    a perfect match in this needs to be reached before can 
    progress with cfpart matching, as the primBuffer provides the
    index to which parts are constituents of each primitive
    """
    assert np.all(pa == pb)

pass
#cfprim(pa,pb,xb,ma)


def cfpart(a, b):
    """
    comparing part buffers (aka csg nodes) 

    1. typecode CSG_UNION/CSG_SPHERE/.. of each part (aka node)  
    2. complement bit and global transform index 
    3. part parameter values 

    """
    assert len(a) == len(b)
    assert a.shape == b.shape
    count = 0 
    cut = 0.0005
    for i in range(len(a)):
        tca = a[i].view(np.int32)[2][3]
        tcb = b[i].view(np.int32)[2][3]
        assert tca == tcb
        if tca != tcb:
            print " tc mismatch %d %d " % (tca, tcb)
        pass
        tc = tca 
        tcn = CSG_.desc(tc)

        # check the complements, viewing as float otherwise lost "-0. ok but not -0"
        coa = np.signbit(a[i].view(np.float32)[3,3])  
        cob = np.signbit(b[i].view(np.float32)[3,3])
        assert coa == cob 

        # shift everything away, leaving just the signbit 
        coa2 = a[i,3,3].view(np.uint32) >> 31
        cob2 = b[i,3,3].view(np.uint32) >> 31
        assert coa2 == cob2 and coa2 == coa and cob2 == cob

        # recover the gtransform index by getting rid of the complement signbit  
        gta = a[i,3,3].view(np.int32) & 0x7fffffff
        gtb = b[i,3,3].view(np.int32) & 0x7fffffff
        assert gta == gtb
        msg = " gt mismatch " if gta != gtb else ""

        if gta < 0 or gtb < 0: msg += " : gta/gtb -ve " 

        mx = np.max(a[i]-b[i])

        if mx > cut or msg != "":
            count += 1 
            print " partIdx:%6d count:%6d tc:%3d tcn:%20s gta:%2d gtb:%2d mx:%10s %s  " % ( i, count, tc, tcn, gta, gtb, mx, msg  )
            #print np.hstack([a[i],b[i]])
        pass
    pass
    print " num_parts %5d  num_discrepant : %5d   cut:%s  " % ( len(a), count, cut  ) 
pass


# boundaries differ due to lack of surfaces in the test, so scrub that  
# as it hides other problems
b.view(np.int32)[:,1,2] = a.view(np.int32)[:,1,2]

cfpart(a,b)






def cftran(ta, tb, cut=0.1):
    """
    comparing tranbuffers
    """
    assert ta.shape == tb.shape
    for i in range(len(ta)):
        mx = np.max(ta[i] - tb[i])
        if mx > cut:
            print i, mx
        pass
    pass






EOP
}



