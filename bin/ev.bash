ev-source(){ echo $BASH_SOURCE ; }
ev-vi(){ vi $(ev-source)  ; }
ev-env(){  olocal- ; opticks- ; }
ev-usage(){ cat << EOU

ev : aiming for flexible/minimal initial event comparison
============================================================

* NB keep it simple, this is for initial comparisons only, 
  for specialized comparisons see opticks/ana

EOU
}

ev-tmp(){   echo  /tmp/$USER/opticks/ev ; }
ev-base(){  echo  $(local-base)/opticks/evt ; }
ev-cd(){   cd $(ev-base) ; }

ev-a-tag(){ echo J ; }
ev-b-tag(){ echo E ; }

ev-get-(){
   local tag=$1
   local cmd 
   if [ "$tag" == "$NODE_TAG" ] ; then
       cmd="rsync -av $TMP/evt/ $tag/"
   else 
       cmd="rsync -av $tag:$TMP/evt/ $tag/"
   fi 
   echo $cmd
   eval $cmd
}

ev-get(){  
    local dir=$(ev-base)
    mkdir -p $dir && cd $dir
    ev-get- $(ev-a-tag)
    ev-get- $(ev-b-tag)
}

ev-mode-notes(){ cat << EON

Currently the interop-OptiX-5 issue forces use of compute mode, 
eg with::

   OKTest --compute --save 
   tboolean-box  --compute     ## save is on by default
   tboolean-cone --compute     ## save is on by default

OKTest 
    ambitious full DYB geometry with central torch source, 
   
    * NB this is triangulated geometry, may be different ?
    * seeing macOS/Linux differences

tboolean-box
    * no significant macOS/Linux differences seen

tboolean-cone


EON
}

#ev-mode(){ echo OKTest ; }
ev-mode(){ echo tboolean-box ; }


ev-rel(){
   case $(ev-mode) in
            OKTest) echo dayabay/torch/1      ;;
      tboolean-box) echo tboolean-box/torch/1 ;; 
   esac

}
ev-a-dir(){ echo $(ev-a-tag)/$(ev-rel) ; }
ev-b-dir(){ echo $(ev-b-tag)/$(ev-rel) ; }

ev-a-(){ echo $(ev-base)/$(ev-a-dir) ; }
ev-b-(){ echo $(ev-base)/$(ev-b-dir) ; }

ev-info(){ cat << EOI

    ev-mode : $(ev-mode)
    ev-rel  : $(ev-rel)

    ev-a-   : $(ev-a-)
    ev-b-   : $(ev-b-)

EOI
}

ev-a(){  cd $(ev-a-); }
ev-b(){  cd $(ev-b-); }

ev-diff()
{  
   ev-cd
   diff -y $(ev-a-dir)/report.txt $(ev-b-dir)/report.txt    

   ev-a 
   np.py
   md5 *

   ev-b
   np.py
   md5 *
}

ev-ox-(){ cat << EOP
import numpy as np
from opticks.ana.hismask import HisMask
hmk = HisMask()

a = np.load("$(ev-a-)/ox.npy")
b = np.load("$(ev-b-)/ox.npy")
assert a.shape == b.shape

ah = a[:,3,3].view(np.uint32)
bh = b[:,3,3].view(np.uint32)
wh = np.where( ah != bh )
abwh = np.vstack( [ ah[wh], bh[wh] ] ).T

sli = slice(0,1000,100)
print "dumping history masks : %s  " % repr(sli)
for ha, hb in zip(ah[sli],bh[sli]):
    print " %20s : %20s " % ( hmk.label(int(ha)), hmk.label(int(hb)) )

print "discrepant history masks " 
for ha, hb in abwh:
    print " %20s : %20s " % ( hmk.label(int(ha)), hmk.label(int(hb)) )
pass

num = len(wh[0])
den = len(ah)
print "%d/%d = %5.3f fraction with different history mask " % (num, den, float(num)/float(den)) 
print abwh

cut = 0.0001
ab = np.max( a[:,:3] - b[:,:3] , axis=(1,2))   ## avoid flags in the comparison (because of NaN from -ve ints viewed as floats)
wab = np.where( np.abs(ab) > cut )[0]
num = len(wab)
print "%d/%d = %5.3f fraction of different final photons (ox), cut %7.4f " % (num, den, float(num)/float(den), cut)   

np.set_printoptions(threshold=2000)
print np.dstack( [ a[wab], b[wab] ] )   


i = a[:,3].view(np.int32)
j = b[:,3].view(np.int32)
ij = np.max( i - j , axis=(1) )
wij = np.where( ij != 0 )[0] 
num = len(wij)
print "%d/%d = %5.3f fraction of different final photon flags (ox) " % (num, den, float(num)/float(den))   

EOP
}

ev-ph-(){ cat << EOP
import numpy as np
from opticks.ana.histype import HisType
hty = HisType()

a = np.load("$(ev-a-)/ph.npy")
b = np.load("$(ev-b-)/ph.npy")
assert a.shape == b.shape

ah = a[:,0].view(np.uint64)
bh = b[:,0].view(np.uint64)
wh = np.where( ah != bh )
abwh = np.vstack( [ ah[wh], bh[wh] ] ).T
print abwh

for ha, hb in abwh:
    print " %50s : %50s " % ( hty.label(int(ha)), hty.label(int(hb)) )
pass

num = len(wh[0])
den = len(ah)
print "%d/%d = %5.3f fraction with different history sequence " % (num, den, float(num)/float(den)) 

EOP
}


ev-genrun(){
    local func=$1
    mkdir -p $(ev-tmp)
    local py=$(ev-tmp)/$func.py
    $func- $* > $py 
    cat $py 
    ipython -i $py 
}
ev-ox(){ ev-genrun $FUNCNAME ; } 
ev-ph(){ ev-genrun $FUNCNAME ; } 




