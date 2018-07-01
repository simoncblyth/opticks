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



ab-i-(){ cat << EOP
import numpy as np

from opticks.ana.mesh import Mesh

a = np.load("$(ab-a-)/partBuffer.npy")
ta = np.load("$(ab-a-)/tranBuffer.npy")
pa = np.load("$(ab-a-)/primBuffer.npy")

b = np.load("$(ab-b-)/partBuffer.npy")
tb = np.load("$(ab-b-)/tranBuffer.npy")
pb = np.load("$(ab-b-)/primBuffer.npy")
xb = np.load("$(ab-b-)/idxBuffer.npy")

mb = Mesh.make("$(ab-b-idpath)")


def cf(a, b):
    assert len(a) == len(b)
    assert a.shape == b.shape
    for i in range(len(a)):
        tca = a[i].view(np.int32)[2][3]
        tcb = b[i].view(np.int32)[2][3]
        tc = tca 
        assert tca == tcb
        if tca != tcb:
            print " tc mismatch %d %d " % (tca, tcb)
        pass

        gta = a[i].view(np.int32)[3][3]
        gtb = b[i].view(np.int32)[3][3]
        assert gta == gtb
        msg = " gt mismatch " if gta != gtb else ""

        mx = np.max(a[i]-b[i])
        print " i:%3d tc:%3d gta:%2d gtb:%2d mx:%10s %s  " % ( i, tc, gta, gtb, mx, msg  )
        if mx > 0.:
            print (a[i]-b[i])/mx
        pass
    pass
pass
#cf(a,b)

w = np.where( pa[:,1] != pb[:,1] )[0]

lv = np.unique(xb[w][:,2])

print "\n".join(map(lambda _:mb.idx2name[_], lv ))



EOP
}

ab-prim(){
   prim.py $(ab-a-)
   prim.py $(ab-b-)
}

ab-i(){ 
   mkdir -p $(ab-tmp)
   local py=$(ab-tmp)/i.py
   ab-i- $* > $py 
   cat $py
   ipython -i $py 
}



