#!/bin/bash -l 

usage(){ cat << EOU

epsilon:tests blyth$ ./U4Material.sh 
Fold : symbol f base /tmp/blyth/opticks/ntds3/G4CXOpticks/stree/mtfold/LS 
f

CMDLINE:/Users/blyth/opticks/u4/tests/U4Material.py
f.base:/tmp/blyth/opticks/ntds3/G4CXOpticks/stree/mtfold/LS

  : f.ABSLENGTH                                        :             (497, 2) : 0:12:41.171411 
  : f.NeutronCONSTANT                                  :               (4, 2) : 0:12:41.158349 
  : f.GROUPVEL                                         :              (18, 2) : 0:12:41.170518 
  : f.PPOTIMECONSTANT                                  :               (2, 2) : 0:12:41.155483 
  : f.RINDEX                                           :              (18, 2) : 0:12:41.154427 
  : f.bisMSBTIMECONSTANT                               :               (2, 2) : 0:12:40.847398 
  : f.SLOWCOMPONENT                                    :             (275, 2) : 0:12:41.153962 
  : f.REEMISSIONPROB                                   :              (28, 2) : 0:12:41.154802 
  : f.bisMSBCOMPONENT                                  :             (275, 2) : 0:12:40.848709 
  : f.bisMSBABSLENGTH                                  :             (375, 2) : 0:12:40.849485 
  : f.RAYLEIGH                                         :              (11, 2) : 0:12:41.155147 
  : f.FASTCOMPONENT                                    :             (275, 2) : 0:12:41.170764 
  : f.GammaCONSTANT                                    :               (4, 2) : 0:12:41.170187 
  : f.NPFold_index                                     :                   19 : 0:12:41.158883 
  : f.bisMSBREEMISSIONPROB                             :              (23, 2) : 0:12:40.848176 
  : f.PPOABSLENGTH                                     :             (770, 2) : 0:12:41.156809 
  : f.PPOCOMPONENT                                     :             (200, 2) : 0:12:41.156393 
  : f.PPOREEMISSIONPROB                                :              (15, 2) : 0:12:41.155945 
  : f.OpticalCONSTANT                                  :               (1, 2) : 0:12:41.157786 
  : f.AlphaCONSTANT                                    :               (4, 2) : 0:12:41.171106 



WOW : LS/PPOABSLENGTH resolution at 0.5nm in some regions 

In [9]: 1240*1e-6/f.PPOABSLENGTH[:,0][::-1]                                                                                                                                
Out[9]: 
array([ 80.   , 189.884, 191.331, 192.78 , 194.23 , 195.048, 195.463, 196.082, 196.913, 197.531, 198.349, 198.766, 199.181, 199.598, 200.213, 200.832, 201.246, 201.449, 201.866, 202.066, 202.896,
       203.312, 203.516, 204.347, 204.749, 205.366, 205.782, 206.398, 207.434, 208.064, 208.68 , 209.498, 209.913, 210.533, 211.15 , 211.564, 212.183, 212.799, 213.834, 215.08 , 215.697, 216.314,
       216.931, 217.551, 218.179, 218.795, 219.213, 219.831, 220.648, 221.267, 222.099, 222.513, 223.13 , 223.75 , 224.365, 224.78 , 225.615, 226.232, 226.633, 227.264, 227.883, 228.084, 228.5  ,
       228.913, 229.328, 229.949, 230.565, 230.981, 231.382, 232.014, 232.419, 232.632, 233.249, 233.662, 234.281, 234.697, 235.098, 235.518, 236.132, 236.551, 236.967, 237.38 , 237.785, 238.2  ,



EOU
}

export FOLD=/tmp/$USER/opticks/ntds3/G4CXOpticks/stree/mtfold/LS
${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/U4Material.py 


