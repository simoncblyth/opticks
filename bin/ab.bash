ab-source(){ echo $BASH_SOURCE ; }
ab-vi(){ vi $(ab-source)  ; }
ab-env(){  olocal- ; opticks- ; }
ab-usage(){ cat << EOU

ab Usage 
===================

ab-i 
    cfprim, cfpart, cftran simple direct buffer comparisons using python 
    instance wrappers 

ab-t
    closer look at discrepant tr using Dir.where_discrepant_tr


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
ab-part(){ ab-genrun $FUNCNAME ; }
ab-plan(){ ab-genrun $FUNCNAME ; }


ab-p-notes(){ cat << EON

For DYB gives 4 discrepant for the two iav + oav, 
(lack of nudging)

See NNodeNudger::

    if(NudgeBuffer) 
         NudgeBuffer->add(root->treeidx, prim.size(), coincidence.size(), nudges.size() );



EON
}


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

xb = b_load("idxBuffer.npy")   ## identity indices for each prim 0?,soIdx,lvIdx,height 
assert len(pa) == len(xb)

uheight = np.unique(xb[:,3])    ## unique tree heights eg [0,1,2,3,4]
max_uheight = uheight.max()
assert max_uheight <= 4, (uheight, max_uheight) 


ma = Mesh.make(a_idpath)
nn = np.load(os.path.expandvars("$TMP/NNodeNudger.npy"))
assert np.all( np.sort(nn[:,0]) == np.arange(len(nn), dtype=np.uint32) )
nns = nn[np.argsort(nn[:,0])]  # sort the nudge info into lvIdx order

log.info(" NNodeNudger nn.shape [%s] per-solid qty expect (num_lv,4)" % (repr(nn.shape)))

print "nudged (lvIdx/treeidx,num_prim,coincidences,nudges)\n nn[np.where( nn[:,3] > 0 )] \n", nn[np.where( nn[:,3] > 0 )]

da = Dir(a_dir)
db = Dir(b_dir)

cut = 0.1
where_discrepant = da.where_discrepant_prims(db, cut)   ## compares param values for all constituent parts of the prim 
wd = np.array(where_discrepant, dtype=np.uint32)        ## primIdx of discrepancies
lvd = np.unique(xb[wd][:,2])    

log.info(" num_discrepant %d cut %s " % ( len(where_discrepant), cut ) )

for detail in [False, True, False]:
    for i in where_discrepant:
        primIdx = i 
        _,soIdx,lvIdx,height = xb[i]
        assert _ == 0   # why is this always zero ?
        name = ma.idx2name[lvIdx]
        nnsi = nns[lvIdx]

        print " %s primIdx:%3d soIdx:%3d lvIdx:%3d height:%d name:%s nnsi:%s   %s " % ( "-" * 5, primIdx, soIdx,lvIdx,height, name, repr(nnsi),  "-" * 10 )

        if detail == True:
            dap = da.prims[i]
            dbp = db.prims[i]

            print "dap.maxdiff(dbp):%s " % dap.maxdiff(dbp)
            print "dap:",dap
            print "dbp:",dbp
            print
            print
        pass
    pass
pass

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


ab-t-notes(){ cat << EON

::

    In [15]: dt = np.max( np.abs(ta[:,0]-tb[:,0]), axis=(1,2))

    In [17]: np.where( dt > 1e-6 )
    Out[17]: (array([], dtype=int64),)

    In [21]: np.where( dt > 1e-7 )[0].shape
    Out[21]: (260,)


EON
}

ab-t-notes(){ cat << EON

[2018-08-01 12:49:31,643] p32697 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 
[2018-08-01 12:49:32,675] p32697 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 
[2018-08-01 12:49:33,732] p32697 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
[2018-08-01 12:49:33,733] p32697 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
 prim with discrepant_tr 0 cut 0.1 
 prim with discrepant_tr 0 cut 0.01 
 prim with discrepant_tr 0 cut 0.001 
 prim with discrepant_tr 0 cut 1e-06 
 prim with discrepant_tr 156 cut 1e-07 
 ------------------------------ primIdx:283 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
 ------------------------------ primIdx:285 soIdx: 67 lvIdx: 56 height:4 name:RadialShieldUnit0xc3d7da8  ------------------------------------------------------------ 
 ...

EON
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

for cut in [0.1,0.01,0.001,1e-6, 1e-7]:
    where_discrepant_tr = da.where_discrepant_tr(db, cut) 
    wd = np.array(where_discrepant_tr, dtype=np.uint32)
    lvd = np.unique(xb[wd][:,2])
    print " prim with discrepant_tr %d cut %s " % ( len(where_discrepant_tr), cut ) 
pass


detail = False

lvs = set()

for i in where_discrepant_tr:

    primIdx = i 
    _,soIdx,lvIdx,height = xb[i]
    name = ma.idx2name[lvIdx]

    assert _ == 0   # why is this always zero ?

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


ab-i-notes(){ cat << EON

cfpart
---------

After absolution of the diffs, get 4 cones showing up as well as the 4 cylinders.

::

    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-i.py
    [2018-07-31 23:53:10,696] p30680 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
    [2018-07-31 23:53:11,668] p30680 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
    [2018-07-31 23:53:12,641] p30680 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
    [2018-07-31 23:53:12,641] p30680 {/Users/blyth/opticks/ana/mesh.py:37} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
     partIdx:     4 count:     1 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     partIdx:    18 count:     3 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     partIdx:  3496 count:     5 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   
     partIdx:  3510 count:     7 tc: 12 tcn:            cylinder gta: 1 gtb: 1 mx:       1.0   


     partIdx:    15 count:     2 tc: 15 tcn:                cone gta: 2 gtb: 2 mx:       1.0   
     partIdx:    29 count:     4 tc: 15 tcn:                cone gta: 2 gtb: 2 mx:       1.0   
     partIdx:  3507 count:     6 tc: 15 tcn:                cone gta: 2 gtb: 2 mx:       1.0   
     partIdx:  3521 count:     8 tc: 15 tcn:                cone gta: 2 gtb: 2 mx:       1.0   


     num_parts 11984  num_discrepant :     8   cut:0.0005  



All the big diffs are 1mm, presumably from nudge not being applied::

    In [5]: ab = np.max( np.abs( a - b  ), axis=(1,2) )

    In [7]: np.where( ab > 1e-3 )
    Out[7]: (array([   4,   15,   18,   29, 3496, 3507, 3510, 3521]),)

    In [8]: w = np.where( ab > 1e-3 )

    In [11]: a[w] - b[w]
    Out[11]: 
    array([[[ 0.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]],

           [[ 0., -1.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]],






EON
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
    :param pa: prim buffer a
    :param pb: prim buffer b
    :param xb: idx buffer b
    :param ma: mesh info used for name lookups

    high level comparison of prim buffers

    primBuffer will be matched when all prim trees have same heights
    and the usage of tranforms and planes within each prim are the same

    a perfect match in this needs to be reached before can 
    progress with cfpart matching, as the primBuffer provides the
    index to which parts are constituents of each primitive
    """
    log.info("cfprim checking identical prim buffers")
    assert np.all(pa == pb)
pass
cfprim(pa,pb,xb,ma)


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
    log.info("cfpart(a,b) a:%s b:%s cut:%s " % (repr(a.shape), repr(b.shape), cut))

    for i in range(len(a)):
        tca = a[i].view(np.int32)[2][3]   ## typecode of node
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

        ab = np.abs(a[i]-b[i])
        mx = np.max(ab)

        if mx > cut or msg != "":
            count += 1 
            print " partIdx:%6d count:%6d tc:%3d tcn:%20s gta:%2d gtb:%2d mx:%10s %s  " % ( i, count, tc, tcn, gta, gtb, mx, msg  )
            #print np.hstack([a[i],b[i],ab])
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

    Note the t,v,q triplet transforms: transform, inverse, inverse-trasposed

    1. Have to cut at 1e-7 (float precision level) to see deviations on the transform
    2. deviations at 0.1 level are apparent on the inverse and inverse transposed 
       (assuming python/C++ numerical difference)

    """
    assert ta.shape == tb.shape
    assert ta.shape[1:] == (3,4,4) 
    assert tb.shape[1:] == (3,4,4) 

    log.info("cftran(ta,tb)  ta:%s tb:%s cut:%s " % (repr(ta.shape), repr(tb.shape), cut))

    count = {}
    for j in [0,1,2]: 
        count[j] = 0 
        for i in range(len(ta)):
            mx = np.max(np.abs(ta[i,j] - tb[i,j]))
            if mx > cut:
                count[j] += 1 
                #print i,j,mx
            pass
        pass
        log.info("cftran j:%d (t,v,q) number of transform deviations %s exceeding cut : %s " % (j,count[j], cut) )
    pass
pass
cftran(ta, tb, 1e-4)
cftran(ta, tb, 0.1)

EOP
}



ab-part-notes(){ cat << EON

Differences in CSG constitutent part params at a level that can be ignored.

args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-part.py
[2018-08-01 20:36:32,863] p78152 {/tmp/blyth/opticks/bin/ab/ab-part.py:22} INFO -  ab.max() 0.00024414062 
[2018-08-01 20:36:32,863] p78152 {/tmp/blyth/opticks/bin/ab/ab-part.py:28} INFO -  part deviation > cut        0.1 :  0/11984 
[2018-08-01 20:36:32,863] p78152 {/tmp/blyth/opticks/bin/ab/ab-part.py:28} INFO -  part deviation > cut       0.01 :  0/11984 
[2018-08-01 20:36:32,863] p78152 {/tmp/blyth/opticks/bin/ab/ab-part.py:28} INFO -  part deviation > cut      0.001 :  0/11984 
[2018-08-01 20:36:32,863] p78152 {/tmp/blyth/opticks/bin/ab/ab-part.py:28} INFO -  part deviation > cut     0.0001 :  80/11984 
[2018-08-01 20:36:32,863] p78152 {/tmp/blyth/opticks/bin/ab/ab-part.py:28} INFO -  part deviation > cut      1e-05 :  80/11984 
[2018-08-01 20:36:32,863] p78152 {/tmp/blyth/opticks/bin/ab/ab-part.py:28} INFO -  part deviation > cut      1e-06 :  82/11984 
[2018-08-01 20:36:32,863] p78152 {/tmp/blyth/opticks/bin/ab/ab-part.py:28} INFO -  part deviation > cut      1e-07 :  82/11984 
[2018-08-01 20:36:32,864] p78152 {/tmp/blyth/opticks/bin/ab/ab-part.py:28} INFO -  part deviation > cut      1e-08 :  82/11984 
[2018-08-01 20:36:32,864] p78152 {/tmp/blyth/opticks/bin/ab/ab-part.py:28} INFO -  part deviation > cut      1e-09 :  82/11984 
[2018-08-01 20:36:32,864] p78152 {/tmp/blyth/opticks/bin/ab/ab-part.py:28} INFO -  part deviation > cut      1e-10 :  82/11984 

EON
}

ab-part-(){ cat << EOP

import os, logging, numpy as np
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

a_dir = "$(ab-a-)"
b_dir = "$(ab-b-)"

a_load = lambda _:np.load(os.path.join(a_dir, _))
b_load = lambda _:np.load(os.path.join(b_dir, _))

a = a_load("partBuffer.npy")
b = b_load("partBuffer.npy")

scrub_boundary = True 
if scrub_boundary:
    b.view(np.int32)[:,1,2] = a.view(np.int32)[:,1,2]
pass

assert a.shape == b.shape
ab = np.max(np.abs(a - b), axis=(1,2))
log.info( " ab.max() %s " % ab.max() )
assert len(ab) == len(a)

for cut in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
    w = np.where( ab > cut )
    n = len(w[0])
    log.info( " part deviation > cut %10s :  %d/%d " % ( cut, n, len(ab) ))
    if n < 25:
        print np.hstack( [a[w]-b[w],a[w], b[w]])
    pass
pass

EOP

}


ab-plan-notes(){ cat << EON

Deviations look like float level precision on some planes at large distances from origin. 

args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-plan.py
[2018-08-01 20:32:39,271] p78125 {/tmp/blyth/opticks/bin/ab/ab-plan.py:19} INFO -  ab.max() 0.15625 
[2018-08-01 20:32:39,272] p78125 {/tmp/blyth/opticks/bin/ab/ab-plan.py:24} INFO -  plan deviation > cut        0.1 :  20/672 
[[     -0.          -0.           0.           0.1562       0.8231      -0.5678       0.      442361.62         0.8231      -0.5678       0.      442361.47  ]
 [      0.          -0.           0.           0.125        0.5678       0.8231       0.     -665388.9          0.5678       0.8231       0.     -665389.    ]
 [     -0.          -0.           0.           0.1562       0.8231      -0.5678       0.      442361.62         0.8231      -0.5678       0.      442361.47  ]
 [      0.          -0.           0.           0.125        0.5678       0.8231       0.     -665388.9          0.5678       0.8231       0.     -665389.    ]
 [     -0.          -0.           0.           0.1562       0.8231      -0.5678       0.      442361.62         0.8231      -0.5678       0.      442361.47  ]
 [      0.          -0.           0.           0.125        0.5678       0.8231       0.     -665388.9          0.5678       0.8231       0.     -665389.    ]
 [     -0.          -0.           0.           0.1562       0.8231      -0.5678       0.      442361.62         0.8231      -0.5678       0.      442361.47  ]
 [      0.          -0.           0.           0.125        0.5678       0.8231       0.     -665388.9          0.5678       0.8231       0.     -665389.    ]
 [     -0.          -0.           0.           0.1562       0.8231      -0.5678       0.      447666.56         0.8231      -0.5678       0.      447666.4   ]
 [      0.           0.           0.          -0.125       -0.5405       0.8413       0.     -668792.8         -0.5405       0.8413       0.     -668792.7   ]
 [      0.          -0.           0.           0.1094       0.9836       0.1805       0.     -156807.05         0.9836       0.1805       0.     -156807.16  ]
 [     -0.          -0.           0.           0.1562       0.8231      -0.5678       0.      447666.56         0.8231      -0.5678       0.      447666.4   ]
 [      0.           0.           0.          -0.125       -0.5405       0.8413       0.     -668792.8         -0.5405       0.8413       0.     -668792.7   ]
 [      0.          -0.           0.           0.1094       0.9836       0.1805       0.     -156807.05         0.9836       0.1805       0.     -156807.16  ]
 [     -0.          -0.           0.           0.1562       0.8231      -0.5678       0.      447666.56         0.8231      -0.5678       0.      447666.4   ]
 [      0.           0.           0.          -0.125       -0.5405       0.8413       0.     -668792.8         -0.5405       0.8413       0.     -668792.7   ]
 [      0.          -0.           0.           0.1094       0.9836       0.1805       0.     -156807.05         0.9836       0.1805       0.     -156807.16  ]
 [     -0.          -0.           0.           0.1562       0.8231      -0.5678       0.      447666.56         0.8231      -0.5678       0.      447666.4   ]
 [      0.           0.           0.          -0.125       -0.5405       0.8413       0.     -668792.8         -0.5405       0.8413       0.     -668792.7   ]
 [      0.          -0.           0.           0.1094       0.9836       0.1805       0.     -156807.05         0.9836       0.1805       0.     -156807.16  ]]
[2018-08-01 20:32:39,274] p78125 {/tmp/blyth/opticks/bin/ab/ab-plan.py:24} INFO -  plan deviation > cut       0.01 :  114/672 
[2018-08-01 20:32:39,274] p78125 {/tmp/blyth/opticks/bin/ab/ab-plan.py:24} INFO -  plan deviation > cut      0.001 :  133/672 
[2018-08-01 20:32:39,275] p78125 {/tmp/blyth/opticks/bin/ab/ab-plan.py:24} INFO -  plan deviation > cut     0.0001 :  135/672 
[2018-08-01 20:32:39,275] p78125 {/tmp/blyth/opticks/bin/ab/ab-plan.py:24} INFO -  plan deviation > cut      1e-05 :  135/672 
[2018-08-01 20:32:39,275] p78125 {/tmp/blyth/opticks/bin/ab/ab-plan.py:24} INFO -  plan deviation > cut      1e-06 :  135/672 
[2018-08-01 20:32:39,275] p78125 {/tmp/blyth/opticks/bin/ab/ab-plan.py:24} INFO -  plan deviation > cut      1e-07 :  161/672 
[2018-08-01 20:32:39,275] p78125 {/tmp/blyth/opticks/bin/ab/ab-plan.py:24} INFO -  plan deviation > cut      1e-08 :  240/672 
[2018-08-01 20:32:39,275] p78125 {/tmp/blyth/opticks/bin/ab/ab-plan.py:24} INFO -  plan deviation > cut      1e-09 :  240/672 
[2018-08-01 20:32:39,275] p78125 {/tmp/blyth/opticks/bin/ab/ab-plan.py:24} INFO -  plan deviation > cut      1e-10 :  240/672 


EON
}

ab-plan-(){ cat << EOP

import os, logging, numpy as np
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

a_dir = "$(ab-a-)"
b_dir = "$(ab-b-)"

a_load = lambda _:np.load(os.path.join(a_dir, _))
b_load = lambda _:np.load(os.path.join(b_dir, _))

a = a_load("planBuffer.npy")
b = b_load("planBuffer.npy")

assert a.shape == b.shape
ab = np.max(np.abs(a - b), axis=(1))

assert len(ab) == len(a)
log.info( " ab.max() %s " % ab.max() )

for cut in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
    w = np.where( ab > cut )
    n = len(w[0])
    log.info( " plan deviation > cut %10s :  %d/%d " % ( cut, n, len(ab) ))
    if n < 25:
        print np.hstack( [a[w]-b[w],a[w], b[w]])
    pass
pass


EOP

}



