#!/usr/bin/env python
"""


::

     245         //  X4PhysicalVolume::convertNode
     246 
     247         LOG(LEVEL) << "[ GParts::Make i " << i << " lvIdx " << lvIdx << " ndIdx " << ndIdx ;
     248         GParts* parts = GParts::Make( csg, spec.c_str(), ndIdx );
     249         LOG(LEVEL) << "] GParts::Make i " << i << " lvIdx " << lvIdx << " ndIdx " << ndIdx ;
     250 
     251         unsigned num_mismatch = 0 ;
     252         parts->applyPlacementTransform( placement, verbosity, num_mismatch );
     253         if(num_mismatch > 0 )
     254         {
     255             LOG(error) << " pt " << i << " invert_trs num_mismatch : " << num_mismatch ;
     256             mismatch_pt.push_back(i);
     257             if(mismatch_placements)
     258             {
     259                 glm::mat4 placement_(placement);
     260                 placement_[0][3] = SPack::unsigned_as_float(i);
     261                 placement_[1][3] = SPack::unsigned_as_float(lvIdx);
     262                 placement_[2][3] = SPack::unsigned_as_float(ndIdx);
     263                 placement_[3][3] = SPack::unsigned_as_float(num_mismatch);
     264                 mismatch_placements->push_back(placement_);
     265             }
     266         }
     267         com->add( parts );
     268     }

    In [18]: i                                                                                                                                                                                                
    Out[18]: 
    array([[  944,    39, 66464,     3],
           [  951,    39, 66471,     3],
           [  959,    39, 66479,     3],
           [  966,    39, 66486,     3],
           [ 1040,    42, 66560,     3],
           [ 1055,    42, 66575,     3],
           [ 1071,    43, 66591,     3],
           [ 1086,    43, 66606,     3],
           [ 1251,    49, 66771,     3],
           [ 1266,    49, 66786,     3],
           [ 1276,    50, 66796,     3],
           [ 1279,    50, 66799,     3],
           [ 1280,    50, 66800,     3],
           [ 1291,    50, 66811,     3],
           [ 1294,    50, 66814,     3],
           [ 1295,    50, 66815,     3],
           [ 1332,    52, 66852,     3],
           [ 1343,    52, 66863,     3],
           [ 1347,    52, 66867,     3],
           [ 1358,    52, 66878,     3],
           [ 1522,    58, 67042,     3],
           [ 1537,    58, 67057,     3],
           [ 2052,    76, 67572,     3],
           [ 2067,    76, 67587,     3],
           [ 2084,    77, 67604,     3],
           [ 2092,    77, 67612,     3],
           [ 2099,    77, 67619,     3],
           [ 2107,    77, 67627,     3],
           [ 2121,    78, 67641,     3],
           [ 2136,    78, 67656,     3],
           [ 2483,    91, 68003,     1],
           [ 2487,    91, 68007,     1],
           [ 2498,    91, 68018,     1],
           [ 2502,    91, 68022,     1],
           [ 2539,    91, 68059,     1],
           [ 2551,    91, 68071,     1],
           [ 2554,    91, 68074,     1],
           [ 2566,    91, 68086,     1],
           [ 2570,    91, 68090,     1],
           [ 2571,    91, 68091,     1],
           [ 2585,    91, 68105,     1],
           [ 2586,    91, 68106,     1]], dtype=uint32)

    In [20]: np.unique(i[:,1], return_counts=True)                                                                                                                                                            
    Out[20]: 
    (array([39, 42, 43, 49, 50, 52, 58, 76, 77, 78, 91], dtype=uint32),
     array([ 4,  2,  2,  2,  6,  4,  2,  2,  4,  2, 12]))


See ~/jnu/issues/geocache-j2102-shakedown.rst

::

      epsilon:GNodeLib blyth$ cat.py -s 66464,66471,66479,66486,66560,66575,66591,66606,66771,66786,66796,66799,66800,66811,66814,66815,66852,66863,66867,66878,67042,67057,67572,67587,67604,67612,67619,67627,67641,67656,68003,68007,68018,68022,68059,68071,68074,68086,68090,68091,68105,68106 all_volume_PVNames.txt
        66464 66465 GLb2.up07_HBeam_phys0x34a9c20
        66471 66472 GLb2.up07_HBeam_phys0x34aa390
        66479 66480 GLb2.up07_HBeam_phys0x34aac10
        66486 66487 GLb2.up07_HBeam_phys0x34ab380
        66560 66561 GLb1.up04_HBeam_phys0x34b3a00
        66575 66576 GLb1.up04_HBeam_phys0x34b49f0
        66591 66592 GLb1.up03_HBeam_phys0x34b6310
        66606 66607 GLb1.up03_HBeam_phys0x34b7300
        66771 66772 GLb2.bt03_HBeam_phys0x34957f0
        66786 66787 GLb2.bt03_HBeam_phys0x34967e0
        66796 66797 GLb2.bt04_HBeam_phys0x3497aa0
        66799 66800 GLb2.bt04_HBeam_phys0x3497dd0
        66800 66801 GLb2.bt04_HBeam_phys0x3497ee0
        66811 66812 GLb2.bt04_HBeam_phys0x3498a90
        66814 66815 GLb2.bt04_HBeam_phys0x3498dc0
        66815 66816 GLb2.bt04_HBeam_phys0x34cea10
        66852 66853 GLb1.bt06_HBeam_phys0x34d21a0
        66863 66864 GLb1.bt06_HBeam_phys0x34d2d50
        66867 66868 GLb1.bt06_HBeam_phys0x34d3190
        66878 66879 GLb1.bt06_HBeam_phys0x34d3d40
        67042 67043 GZ1.A01_02_HBeam_phys0x34e1c40
        67057 67058 GZ1.A01_02_HBeam_phys0x34e2c30
        67572 67573 ZC2.A03_B04_HBeam_phys0x3515550
        67587 67588 ZC2.A03_B04_HBeam_phys0x3516450
        67604 67605 ZC2.A04_B05_HBeam_phys0x3517d80
        67612 67613 ZC2.A04_B05_HBeam_phys0x3518580
        67619 67620 ZC2.A04_B05_HBeam_phys0x3518c80
        67627 67628 ZC2.A04_B05_HBeam_phys0x3519480
        67641 67642 ZC2.A05_B06_HBeam_phys0x351aab0
        67656 67657 ZC2.A05_B06_HBeam_phys0x351b9b0
        68003 68004 lSteel_phys0x353ac50
        68007 68008 lSteel_phys0x353b010
        68018 68019 lSteel_phys0x353ba60
        68022 68023 lSteel_phys0x353be20
        68059 68060 lSteel_phys0x353e4f0
        68071 68072 lSteel_phys0x353f030
        68074 68075 lSteel_phys0x353f300
        68086 68087 lSteel_phys0x353fe40
        68090 68091 lSteel_phys0x3540200
        68091 68092 lSteel_phys0x35402f0
        68105 68106 lSteel_phys0x3541010
        68106 68107 lSteel_phys0x3541100
        epsilon:GNodeLib blyth$ 



"""
import os, numpy as np
np.set_printoptions(suppress=True, linewidth=200)

if __name__ == '__main__':
     path = os.path.expandvars("/tmp/$USER/opticks/GGeo__deferredCreateGParts/mm0/mismatch_placements.npy")
     a = np.load(path)
     i = a.view(np.uint32)[:,:,3].copy()   

     a[:,0,3] = 0.   # scrub   
     a[:,1,3] = 0.   
     a[:,2,3] = 0.   
     a[:,3,3] = 1.   

     cmd = "cat.py -s " + ",".join(map(str,i[:,2])) + " all_volume_LVNames.txt"   



