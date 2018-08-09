ab-source(){ echo $BASH_SOURCE ; }
ab-vi(){ vi $(ab-source)  ; }
ab-env(){  olocal- ; opticks- ; }
ab-usage(){ cat << EOU

ab Usage 
===================

ab-l
    list cachemeta in the geocache dirs being compared, to check last update time    
ab-ls
    more detailed listing of geocache dirs 

ab-i 
    cfprim, cfpart, cftran simple direct buffer comparisons using python 
    instance wrappers 

ab-t
    closer look at discrepant tr using Dir.where_discrepant_tr


To do a full check of analytic geometry should also look at the 
repeated geometry by changing AB_TAIL : however most issues (deep shapes etc..) 
are with the global geometry in slot 0::

    ab-;AB_TAIL=0 ab-t    
    ##   why is there no 1 ? unconfirmed guess is some repeated RPC geometry that got was cut by geoselection 
    ab-;AB_TAIL=2 ab-t 
    ab-;AB_TAIL=3 ab-t 
    ab-;AB_TAIL=4 ab-t 
    ab-;AB_TAIL=5 ab-t 

EOU
}


ab-base(){  echo  /usr/local/opticks/geocache ; }
ab-cd(){   cd $(ab-base) ; }

ab-tmp(){ echo /tmp/$USER/opticks/bin/ab ; }

ab-a-dir(){ echo DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd  ; }

#ab-b-dir(){ echo OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe ; }
#ab-b-dir(){ echo OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/742ab212f7f2da665ed627411ebdb07d ; }
ab-b-dir(){ echo OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/0dce832a26eb41b58a000497a3127cb8 ; }

ab-A-(){ echo $(ab-base)/$(ab-a-dir)/104 ; }
#ab-A-(){ echo $(ab-base)/$(ab-a-dir)/103 ; }
#ab-A-(){ echo $(ab-base)/$(ab-a-dir)/1 ; }
ab-B-(){ echo $(ab-base)/$(ab-b-dir)/1 ; }

ab-A(){  cd $(ab-A-); }
ab-B(){  cd $(ab-B-); }

ab-tail(){ echo ${AB_TAIL:-0} ; }
ab-a-(){ echo $(ab-A-)/GPartsAnalytic/$(ab-tail) ; }
ab-b-(){ echo $(ab-B-)/GParts/$(ab-tail) ; }

ab-a-tails(){ ls -1 $(ab-b-)/../ ; }
ab-b-tails(){ ls -1 $(ab-b-)/../ ; }

ab-a(){  cd $(ab-a-); }
ab-b(){  cd $(ab-b-); }

ab-a-run(){  OPTICKS_RESOURCE_LAYOUT=104 OKTest -G --gltf 3   ; }
ab-a-lldb(){  OPTICKS_RESOURCE_LAYOUT=104 lldb -- OKTest -G --gltf 3   ; }

ab-b-run(){  OPTICKS_RESOURCE_LAYOUT=104 OKX4Test --x4polyskip 211,232  ; }
ab-b-lldb(){  OPTICKS_RESOURCE_LAYOUT=104 lldb -- OKX4Test --x4polyskip 211,232 ; }


ab-paths(){ cat << EOP

   ab-base  : $(ab-base)
   ab-a-dir : $(ab-a-dir)
   ab-b-dir : $(ab-b-dir)

   ab-A-   : $(ab-A-)
   ab-B-   : $(ab-B-)

   ab-tail : $(ab-tail)
   ab-a-   : $(ab-a-)
   ab-b-   : $(ab-b-)

EOP
}


ab-l(){ 
   date
   echo A $(ls -l $(ab-A-)/*.json)
   echo B $(ls -l $(ab-B-)/*.json)
}
ab-l-notes(){ cat << EON

Check the dates of the cachemeta.json to see when the caches were last written::

    epsilon:npy blyth$ ab-l
    A -rw-r--r-- 1 blyth staff 63 Aug 4 20:15 /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/104/cachemeta.json
    B -rw-r--r-- 1 blyth staff 53 Aug 4 20:46 /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/742ab212f7f2da665ed627411ebdb07d/1/cachemeta.json

EON
}

ab-ls(){ 
   date
   echo A $(ab-A-)
   ls -l $(ab-A-) 
   echo B $(ab-B-)
   ls -l $(ab-B-) 
}

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


ab-prim-a(){ prim.py $(ab-a-) $*; }
ab-prim-b(){ prim.py $(ab-b-) $*; }

ab-prim(){
   prim.py $(ab-a-) $*
   prim.py $(ab-b-) $*
}
ab-prim-notes(){ cat << EON

AB_TAIL=5 ab-prim 0:1
    dump the analytic geometry of the first primIdx of instance 5 
    which is pmt-hemi from the two geocache 

EON
}



ab-genrun(){
    local func=$1
    mkdir -p $(ab-tmp)
    local py=$(ab-tmp)/$func.py
    $func- $* > $py 
    cat $py
    ipython -i $py 
}

ab-lv2name(){

   local d
   local dirs="$(ab-A-) $(ab-B-)"

   local rel="MeshIndex/GItemIndexSource.json"
   #local rel="MeshIndex/GItemIndexLocal.json"

   for d in $dirs ; do 
      cd $d
      pwd
      ls -l $rel
      js.py $rel | head -5
      echo ...
      js.py $rel | tail -5 
   done 
}

ab-lv2name-notes(){ cat << EON

MeshIndex lvIdx to lvName mapping is totally off for live geometry 
====================================================================

* FIXED in X4PhysicalVolume by reworking convertSolids into postorder together with lvIdx

Noticed this from getting incorrect lvNames from ab-prim when 
it was using IDPATH from the direct geocache. 

Investigate in :doc:`/notes/issues/ab-lvname`  



::

    epsilon:1 blyth$ ab-;ab-lv2name
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/104
    -rw-r--r--  1 blyth  staff  9448 Aug  7 13:18 MeshIndex/GItemIndexSource.json
      0 : near_top_cover_box0xc23f970 
      1 : RPCStrip0xc04bcb0 
      2 : RPCGasgap140xbf4c660 
      3 : RPCBarCham140xc2ba760 
      4 : RPCGasgap230xbf50468 
    ...
    244 : near-radslab-box-80xcd308c0 
    245 : near-radslab-box-90xcd31ea0 
    246 : near_hall_bot0xbf3d718 
    247 : near_rock0xc04ba08 
    248 : WorldBox0xc15cf40 
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/0dce832a26eb41b58a000497a3127cb8/1
    -rw-r--r--  1 blyth  staff  9472 Aug  7 13:19 MeshIndex/GItemIndexSource.json
      0 : WorldBox0xc15cf40 
      1 : near_rock0xc04ba08 
      2 : near_hall_top_dwarf0xc0316c8 
      3 : near_top_cover_box0xc23f970 
      4 : RPCMod0xc13bfd8 
    ...
    244 : near-radslab-box-50xccefd60 
    245 : near-radslab-box-60xccefda0 
    246 : near-radslab-box-70xccefde0 
    247 : near-radslab-box-80xcd308c0 
    248 : near-radslab-box-90xcd31ea0 
    epsilon:1 blyth$ 


EON
}



ab-p-desc(){ echo "Comparison of param values for all constituent parts of each prim " ; }
ab-p(){ ab-genrun $FUNCNAME ; }
ab-p-(){ cat << EOP
import os, numpy as np
from opticks.ana.mesh import Mesh
from opticks.ana.prim import Dir
from opticks.sysrap.OpticksCSG import CSG_

a_dir = "$(ab-a-)"
b_dir = "$(ab-b-)"
A_dir = "$(ab-A-)"
B_dir = "$(ab-B-)"

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


ma = Mesh.make(A_dir)
nn = np.load(os.path.expandvars("$TMP/NNodeNudger.npy"))
assert np.all( np.sort(nn[:,0]) == np.arange(len(nn), dtype=np.uint32) )
nns = nn[np.argsort(nn[:,0])]  # sort the nudge info into lvIdx order

log.info(" NNodeNudger nn.shape [%s] per-solid qty expect (num_lv,4)" % (repr(nn.shape)))

print "nudged (lvIdx/treeidx,num_prim,coincidences,nudges)\n nn[np.where( nn[:,3] > 0 )] \n", nn[np.where( nn[:,3] > 0 )]

da = Dir(a_dir)
db = Dir(b_dir)

mxdump = 5 

for cut in [0.1, 1e-3, 1e-4]:
    where_discrepant = da.where_discrepant_prims(db, cut)   ## compares param values for all constituent parts of the prim 
    wd = np.array(where_discrepant, dtype=np.uint32)        ## primIdx of discrepancies
    lwd = len(where_discrepant)
    lvd = np.unique(xb[wd][:,2])    
    log.info(" num_discrepant %d cut %s " % ( lwd, cut ) )

    for detail in [False, True, False]:
        for i, primIdx in enumerate(where_discrepant):
            _,soIdx,lvIdx,height = xb[primIdx]
            assert _ == 0   # why is this always zero ?
            name = ma.idx2name[lvIdx]
            nnsi = nns[lvIdx]

            dap = da.prims[primIdx]
            dbp = db.prims[primIdx]
            mxd = dap.maxdiff(dbp)

            print " %s mxd*1e6:%10.3f primIdx:%3d soIdx:%3d lvIdx:%3d height:%d name:%s nnsi:%s   %s " % ( "-" * 5, mxd*1e6, primIdx, soIdx,lvIdx,height, name, repr(nnsi),  "-" * 10 )

            if detail == True and i < mxdump:
                print "dap.maxdiff(dbp):%s " % mxd
                print "dap:",dap
                print "dbp:",dbp
                print
                print
            pass
        pass
        if detail and lwd > mxdump:
            log.warning("detail dumping of many more discrepancies was suppressed lwd %d mxdump %d " % ( lwd, mxdump ))
        pass 
    pass
pass

EOP
}

ab-p-notes(){ cat << EON

For DYB initially gave 8 1mm discrepant for the two iav + oav + gdls + lso, 
fixed by some bug fixes to regard nudging.

See NNodeNudger::

    if(NudgeBuffer) 
         NudgeBuffer->add(root->treeidx, prim.size(), coincidence.size(), nudges.size() );


Following the fix to nudging have to go to 1e-4 level to see discrepancies.


EON
}






ab-s(){ ab-genrun $FUNCNAME ; }
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


ab-t-desc(){ echo "Comparison focussing on transforms " ; }
ab-t(){ ab-genrun $FUNCNAME ; }
ab-t-(){ cat << EOP
import os, numpy as np
from opticks.ana.mesh import Mesh
from opticks.ana.prim import Dir
from opticks.sysrap.OpticksCSG import CSG_

a_dir = "$(ab-a-)"
b_dir = "$(ab-b-)"
da = Dir(a_dir)
db = Dir(b_dir)

A_dir = "$(ab-A-)"
B_dir = "$(ab-B-)"

a_load = lambda _:np.load(os.path.join(a_dir, _))
b_load = lambda _:np.load(os.path.join(b_dir, _))

a = a_load("partBuffer.npy")
ta = a_load("tranBuffer.npy")
pa = a_load("primBuffer.npy")

b = b_load("partBuffer.npy")
tb = b_load("tranBuffer.npy")
pb = b_load("primBuffer.npy")

xb = b_load("idxBuffer.npy")

ma = Mesh.make(A_dir)
mb = Mesh.make(B_dir)

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

ab-t-notes(){ cat << EON

::

    In [15]: dt = np.max( np.abs(ta[:,0]-tb[:,0]), axis=(1,2))

    In [17]: np.where( dt > 1e-6 )
    Out[17]: (array([], dtype=int64),)

    In [21]: np.where( dt > 1e-7 )[0].shape
    Out[21]: (260,)


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






ab-i(){ ab-genrun $FUNCNAME ; }
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

A_dir = "$(ab-A-)"
B_dir = "$(ab-B-)"
xb = b_load("idxBuffer.npy")

ma = Mesh.make(A_dir)
mb = Mesh.make(B_dir)


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

scrub_boundary = False 
if scrub_boundary:
    b.view(np.int32)[:,1,2] = a.view(np.int32)[:,1,2]
pass


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


ab-part(){ ab-genrun $FUNCNAME ; }
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

ia = a.view(np.int32)[:,1,2].copy()
ib = b.view(np.int32)[:,1,2].copy()

scrub_boundary = False
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










ab-plan(){ ab-genrun $FUNCNAME ; }
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


ab-blib-smry(){ MODE=1 ab-blib ; }
ab-blib()
{
   echo "A"
   blib.py $(ab-a-)
   echo "B"
   blib.py $(ab-b-)
}
ab-blib-diff0()
{
   local tmp=$(ab-tmp)/$FUNCNAME
   mkdir -p $tmp

   blib.py $(ab-a-) > $tmp/a.txt 
   blib.py $(ab-b-) > $tmp/b.txt 
   
   local cmd="diff -y $tmp/a.txt $tmp/b.txt --width 200"
   echo $cmd
   eval $cmd
}

ab-blib-diff2()
{
   local tmp=$(ab-tmp)/$FUNCNAME
   mkdir -p $tmp

   MODE=2 blib.py $(ab-a-) > $tmp/a2.txt 
   MODE=2 blib.py $(ab-b-) > $tmp/b2.txt 
   
   local cmd="diff -y $tmp/a2.txt $tmp/b2.txt --width 180"
   echo $cmd
   eval $cmd
}


ab-blib-notes(){ cat << EON

Observations: 

1. same number of materials

2. FIXED : two extra surfaces in B ( lvPmtHemiCathodeSensorSurface, lvHeadonPmtCathodeSensorSurface )

   * these are artificial additions.. for model matching 

     * yes, but thats confusing : they are just being converted like all other surfaces
       and they are there thanks to CDetector::attachSurfaces (GDML fixup)   

   * what was wrong with the old 103 one are comparing against ?


3. B is sometimes duplicating isur/osur but A is not : different border count 

   * think this was a fix to better translate the Geant4 meaning of border (with directionality) vs skin surfaces (without directionality)  
   * probably easier to find the logic to decide (skin/border etc..) and adjust the old way to match  

::

    epsilon:cfg4 blyth$ ab-;ab-blib
    A
     nbnd 122 nmat  38 nsur  46 
      0 : Vacuum///Vacuum 
      1 : Vacuum///Rock 
      2 : Rock///Air 
      3 : Air/NearPoolCoverSurface//PPE 
      4 : Air///Aluminium 
      5 : Aluminium///Foam 
      6 : Foam///Bakelite 
      7 : Bakelite///Air 
      8 : Air///MixGas 
    ...
    119 : OwsWater/NearOutOutPiperSurface//PVC 
    120 : DeadWater/LegInDeadTubSurface//ADTableStainlessSteel 
    121 : Rock///RadRock 

    B
     nbnd 127 nmat  38 nsur  46 
      0 : Vacuum///Vacuum 
      1 : Vacuum///Rock 
      2 : Rock///Air 
      3 : Air/NearPoolCoverSurface/NearPoolCoverSurface/PPE 
      4 : Air///Aluminium 
      5 : Aluminium///Foam 
      6 : Foam///Bakelite 
      7 : Bakelite///Air 
      8 : Air///MixGas 
    ...
    124 : OwsWater/NearOutOutPiperSurface/NearOutOutPiperSurface/PVC 
    125 : DeadWater/LegInDeadTubSurface/LegInDeadTubSurface/ADTableStainlessSteel 
    126 : Rock///RadRock 


4. surface count matching but ORDERING DIFFERS 

::

    epsilon:0 blyth$ diff -y $(ab-A-)/GItemList/GSurfaceLib.txt $(ab-B-)/GItemList/GSurfaceLib.txt

    NearPoolCoverSurface<
    NearDeadLinerSurface    NearDeadLinerSurface
    NearOWSLinerSurface     NearOWSLinerSurface
    NearIWSCurtainSurface   NearIWSCurtainSurface
    SSTWaterSurfaceNear1    SSTWaterSurfaceNear1
    SSTOilSurface           SSTOilSurface
    RSOilSurface<
    ESRAirSurfaceTop        ESRAirSurfaceTop
    ESRAirSurfaceBot        ESRAirSurfaceBot
    AdCableTraySurface<
    SSTWaterSurfaceNear2    SSTWaterSurfaceNear2
                           >NearPoolCoverSurface
                           >RSOilSurface
                           >AdCableTraySurface
    PmtMtTopRingSurface     PmtMtTopRingSurface
    PmtMtBaseRingSurface    PmtMtBaseRingSurface
    PmtMtRib1Surface        PmtMtRib1Surface

* FIXED see notes/issues/surface_ordering.rst

EON

}


ab-surf()
{
   echo "A"
   np.py $(ab-A-)/GSurfaceLib  
   echo "B"
   np.py $(ab-B-)/GSurfaceLib  

   diff -y $(ab-A-)/GItemList/GSurfaceLib.txt $(ab-B-)/GItemList/GSurfaceLib.txt
}


ab-surf1(){ ab-genrun $FUNCNAME ; }
ab-surf1-(){ cat << EOP

import os, logging, numpy as np
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

A_dir = "$(ab-A-)"
B_dir = "$(ab-B-)"

a_npy = lambda _:np.load(os.path.join(A_dir, _))
b_npy = lambda _:np.load(os.path.join(B_dir, _))
a_txt = lambda _:map(str.strip,file(os.path.join(A_dir,_)).readlines())
b_txt = lambda _:map(str.strip,file(os.path.join(B_dir,_)).readlines())

a = a_npy("GSurfaceLib/GSurfaceLib.npy")
b = b_npy("GSurfaceLib/GSurfaceLib.npy")

sa = a_txt("GItemList/GSurfaceLib.txt")
sb = b_txt("GItemList/GSurfaceLib.txt")
assert sa == sb 

print "a.shape : %s " % repr(a.shape)
print "b.shape : %s " % repr(b.shape)

assert a.shape == b.shape

ab = np.abs( a - b )
print "ab.max()", ab.max()
abmx = np.max(ab, axis=(1,2,3))

assert len(abmx) == len(a) == len(sa) 

fac = 1e9

print " abmx*%s : max absolute difference of property values of each surface "  % fac 

print "\n".join([ " %10.3f  : %s " % ( abmx[i]*fac, sa[i] ) for i in range(len(ab))]) 

oa = a_npy("GSurfaceLib/GSurfaceLibOptical.npy")
ob = b_npy("GSurfaceLib/GSurfaceLibOptical.npy")
assert oa.shape == ob.shape


s = "np.hstack( [oa, ob] )"
print s
print eval(s)

assert np.all( oa == ob ) 


EOP
}



ab-surf1-notes(){ cat << EON

Small unexplained differences for some surface properties::

args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-surf1.py
a.shape : (48, 2, 39, 4) 
b.shape : (48, 2, 39, 4) 
ab.max() 5.9604645e-08
 abmx*1000000000.0 : max absolute difference of property values of each surface 

      0.000  : ESRAirSurfaceTop 
      0.000  : ESRAirSurfaceBot 
     ......
      0.000  : SSTWaterSurfaceNear2 
     59.605  : NearIWSCurtainSurface 
     59.605  : NearOWSLinerSurface 
     59.605  : NearDeadLinerSurface 
      0.000  : NearPoolCoverSurface 
      7.451  : RSOilSurface 
      0.000  : AdCableTraySurface 
      0.000  : PmtMtTopRingSurface 
     ....
      0.000  : perfectSpecularSurface 
      0.000  : perfectDiffuseSurface 
     59.605  : lvHeadonPmtCathodeSensorSurface 
     59.605  : lvPmtHemiCathodeSensorSurface 


Difference in optical "value" 

::

    np.hstack( [oa, ob] )
    [[  0   0   0   0   0   0   0   0]
     [  1   0   0   0   1   0   0   0]
     [  2   0   3 100   2   0   3   0]
     [  3   0   3 100   3   0   3   0]
     [  4   0   3 100   4   0   3   0]
     [  5   0   3  20   5   0   3   0]
     [  6   0   3  20   6   0   3   0]
     [  7   0   3  20   7   0   3   0]

EON
}











ab-mat()
{
   echo "A"
   np.py $(ab-A-)/GMaterialLib  
   md5 $(ab-A-)/GMaterialLib/GMaterialLib.npy 
   echo "B"
   np.py $(ab-B-)/GMaterialLib  

   echo
   echo A $(ls -l $(ab-A-)/GMaterialLib/GMaterialLib.npy)
   echo B $(ls -l $(ab-B-)/GMaterialLib/GMaterialLib.npy)
   echo
   md5 $(ab-A-)/GMaterialLib/GMaterialLib.npy 
   md5 $(ab-B-)/GMaterialLib/GMaterialLib.npy 
   echo
   diff -y $(ab-A-)/GItemList/GMaterialLib.txt $(ab-B-)/GItemList/GMaterialLib.txt
   diff  $(ab-A-)/GItemList/GMaterialLib.txt $(ab-B-)/GItemList/GMaterialLib.txt
}


ab-mat1(){ ab-genrun $FUNCNAME ; }
ab-mat1-(){ cat << EOP

import os, logging, numpy as np
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

A_dir = "$(ab-A-)"
B_dir = "$(ab-B-)"

a_npy = lambda _:np.load(os.path.join(A_dir, _))
b_npy = lambda _:np.load(os.path.join(B_dir, _))
a_txt = lambda _:map(str.strip,file(os.path.join(A_dir,_)).readlines())
b_txt = lambda _:map(str.strip,file(os.path.join(B_dir,_)).readlines())

a = a_npy("GMaterialLib/GMaterialLib.npy")
b = b_npy("GMaterialLib/GMaterialLib.npy")
assert a.shape == b.shape

ma = a_txt("GItemList/GMaterialLib.txt")
mb = b_txt("GItemList/GMaterialLib.txt")
assert len(ma) == len(mb)
assert len(ma) == len(a) 

assert ma == mb, "material names must match see notes/issues/surface_ordering.rst "  


print " check max deviations per material, and globally " 

ab0 = np.abs( a[:,0] - b[:,0] )
ab0m = np.max(ab0, axis=(1,2) )
print "ab0m", ab0m
print "ab0m.max()", ab0m.max()

ab1 = np.abs( a[:,1] - b[:,1] )
ab1m = np.max(ab1, axis=(1,2) )
print "ab1m", ab1m
print "ab1m.max()", ab1m.max()


EOP
}
ab-mat1-notes(){ cat << EON

Huh all groupvel stuck at 300 in A, but not B::

    In [42]: np.all( a[:,1,:,0] == 300. )
    Out[42]: True

FIXED : by moving replaceGROUPVEL to be done beforeClose rather than on loading 

Also see small differences on some long abslength, scattering lengths::

    In [23]: ab0 = np.abs( a[:,0] - b[:,0] )

    In [24]: ab0.shape
    Out[24]: (38, 39, 4)

    In [25]: ab0m = np.max(ab0, axis=(1,2) ) ; print ab0m 
    Out[25]: 
    array([0.0156, 0.0156, 0.0156, 0.0156, 0.0002, 0.0059, 0.0059, 0.0059, 0.0059, 0.    , 0.    , 0.    , 0.    , 0.0002, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
           0.    , 0.    , 0.    , 0.0156, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ], dtype=float32)





EON
}






ab-bnd(){ ab-genrun $FUNCNAME ; }
ab-bnd-(){ cat << EOP

import os, logging, numpy as np
from opticks.ana.blib import BLib
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

a_dir = "$(ab-a-)"
b_dir = "$(ab-b-)"

a_load = lambda _:np.load(os.path.join(a_dir, _))
b_load = lambda _:np.load(os.path.join(b_dir, _))

a = a_load("partbuffer.npy")
b = b_load("partbuffer.npy")
assert a.shape == b.shape

ia = a.view(np.int32)[:,1,2].copy()
ib = b.view(np.int32)[:,1,2].copy()

print " ia.min() %d ia.max() %d  len(np.unique(ia)) %d  " % ( ia.min(), ia.max(), len(np.unique(ia)) ) 
print " ib.min() %d ib.max() %d  len(np.unique(ib)) %d  " % ( ib.min(), ib.max(), len(np.unique(ib)) ) 

w = np.where( ia != ib )
n = len(w[0])
log.info( " part.bnd diff :  %d/%d " % ( n, len(a) ))
print np.hstack( [ia[w], ib[w]])


r = np.arange(0,10000,1000, dtype=np.int32)
r[0] = 100 

for i in r:
    s = "np.all( ia[:%(i)s] == ib[:%(i)s] ) " % locals() 
    print s, eval(s) 
pass

s = "np.all( ia == ib ) "
print s, eval(s) 


EOP

}

ab-bnd-notes(){ cat << EON

::


    Goes off the rails at part 8143 

    In [19]: ia.shape
    Out[19]: (11984,)

    In [20]: ib.shape
    Out[20]: (11984,)

    In [18]: np.all( ia[:8142] == ib[:8142] )
    Out[18]: True

    In [17]: np.all( ia[:8143] == ib[:8143] )
    Out[17]: False



    In [12]: np.unique(ia)
    Out[12]: 
    array([ 17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
            56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  81,  82,  84,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
            99, 100, 102, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120], dtype=int32)

    In [13]: np.unique(ib)
    Out[13]: 
    array([ 17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  28,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,
            57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  82,  83,  85,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,
            99, 100, 102, 103, 105, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126], dtype=int32)

EON
}





ab-idx(){ ab-genrun $FUNCNAME ; }
ab-idx-(){ cat << EOP

import os, logging, numpy as np
from opticks.ana.blib import BLib
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

a_dir = "$(ab-a-)"
b_dir = "$(ab-b-)"

a_load = lambda _:np.load(os.path.join(a_dir, _))
b_load = lambda _:np.load(os.path.join(b_dir, _))

a = a_load("idxBuffer.npy")
b = b_load("idxBuffer.npy")
assert a.shape == b.shape

print "a %s\n" % repr(a.shape), a
print "b %s\n" % repr(b.shape), b

assert np.all( a[:,0] == b[:,0] )   ## always zero
assert not np.all( a[:,1] == b[:,1] )
assert np.all( a[:,2] == b[:,2] )
assert np.all( a[:,3] == b[:,3] )

## columns are : index,soIdx,lvIdx,height
## 4th column height always matches 

EOP
}

ab-idx-notes(){ cat << EON

These used to agree, but have change B X4PhysicalVolume 
to do convertSolids into postorder tail resulting in soIdx 
becoming the same as lvIdx.  

::

    In [1]: a
    Out[1]: 
    array([[  0,  30, 192,   0],
           [  0,  31,  94,   0],
           [  0,  32,  90,   0],
           ...,
           [  0, 239, 235,   0],
           [  0, 239, 235,   0],
           [  0, 239, 235,   0]], dtype=uint32)

    In [2]: b
    Out[2]: 
    array([[  0, 192, 192,   0],
           [  0,  94,  94,   0],
           [  0,  90,  90,   0],
           ...,
           [  0, 235, 235,   0],
           [  0, 235, 235,   0],
           [  0, 235, 235,   0]], dtype=uint32)



EON
}


