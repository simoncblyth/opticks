all-volume-identity-tripletid-zero
======================================


Issue : Triplet identities all zero
--------------------------------------

::

    In [1]: a = np.load("all_volume_identity.npy")                                                                                                                                                         

    In [2]: a                                                                                                                                                                                              
    Out[2]: 
    array([[         0,          0,   16252928, 4294967295],
           [         1,          0,   16187393, 4294967295],
           [         2,          0,    1376258, 4294967295],
           ...,
           [     12227,          0,   15925374, 4294967295],
           [     12228,          0,   15990910, 4294967295],
           [     12229,          0,   16056446, 4294967295]], dtype=uint32)

    In [3]: a[:,1].min()                                                                                                                                                                                   
    Out[3]: 0

    In [4]: a[:,1].max()                                                                                                                                                                                   
    Out[4]: 0


After Fix
-----------

::

    In [1]: a = np.load("all_volume_identity.npy")                                                                                                                                                         

    In [2]: a                                                                                                                                                                                              
    Out[2]: 
    array([[         0,          0,   16252928, 4294967295],
           [         1,          1,   16187393, 4294967295],
           [         2,          2,    1376258, 4294967295],
           ...,
           [     12227,       4483,   15925374, 4294967295],
           [     12228,       4484,   15990910, 4294967295],
           [     12229,       4485,   16056446, 4294967295]], dtype=uint32)

    In [3]: a[:,1]                                                                                                                                                                                         
    Out[3]: array([   0,    1,    2, ..., 4483, 4484, 4485], dtype=uint32)

    In [4]: a[:,1].min()                                                                                                                                                                                   
    Out[4]: 0

    In [5]: a[:,1].max()                                                                                                                                                                                   
    Out[5]: 84057860

    In [6]: hex(84057860)                                                                                                                                                                                  
    Out[6]: '0x5029f04'

    In [7]: a.shape                                                                                                                                                                                        
    Out[7]: (12230, 4)

    In [8]: 84057860 >> 24                                                                                                                                                                                 
    Out[8]: 5





Cause
------

GNodeLib::add happening too early... before the labeling::

    frame #4: 0x000000010a8d0c3d libGGeo.dylib`GVolume::getIdentity(this=0x000000011fd174a0) const at GVolume.cc:275
    frame #5: 0x000000010a91eef9 libGGeo.dylib`GNodeLib::add(this=0x0000000113105b70, volume=0x000000011fd174a0) at GNodeLib.cc:270
    frame #6: 0x000000010a916284 libGGeo.dylib`GGeo::add(this=0x0000000116e41160, volume=0x000000011fd174a0) at GGeo.cc:904
    frame #7: 0x000000010363f222 libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfdd98, pv=0x0000000116ebf630, parent=0x000000011fd16ba0, depth=1, parent_pv=0x0000000115800070, recursive_select=0x00007ffeefbfd013) at X4PhysicalVolume.cc:1016
    frame #8: 0x000000010363f27f libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfdd98, pv=0x0000000115800070, parent=0x0000000000000000, depth=0, parent_pv=0x0000000000000000, recursive_select=0x00007ffeefbfd013) at X4PhysicalVolume.cc:1023
    frame #9: 0x000000010363996c libExtG4.dylib`X4PhysicalVolume::convertStructure(this=0x00007ffeefbfdd98) at X4PhysicalVolume.cc:947
    frame #10: 0x00000001036388f3 libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfdd98) at X4PhysicalVolume.cc:201
    frame #11: 0x00000001036385b5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfdd98, ggeo=0x0000000116e41160, top=0x0000000115800070) at X4PhysicalVolume.cc:180
    frame #12: 0x0000000103637875 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfdd98, ggeo=0x0000000116e41160, top=0x0000000115800070) at X4PhysicalVolume.cc:171
    frame #13: 0x000000010001591f OKX4Test`main(argc=15, argv=0x00007ffeefbfe598) at OKX4Test.cc:100


Fix
----

0. remove GGeo::add GVolume from X4PhysicalVolume::convertStructure_r
1. add setRoot/getRoot/m_root to GNodeLib/GGeo and invoke from X4PhysicalVolume::convertStructure
2. add collectNodes to GInstancer for deferred GNodeLib population invoked from GInstancer::createInstancedMergedMeshes



Checking Identity
-----------------

::

    In [16]: tid = a[:,1]                                                                                                                                                                                  

    In [17]: ridx = tid >> 24                                                                                                                                                                              

    In [18]: pidx = np.where( ridx == 0,                       0, ( tid >>  8 ) & 0xffff )                                                                                                                 

    In [19]: oidx = np.where( ridx == 0, ( tid >> 0 ) & 0xffffff, ( tid >> 0  ) & 0xff   )                                                                                                                 

    In [20]: oidx                                                                                                                                                                                          
    Out[20]: array([   0,    1,    2, ..., 4483, 4484, 4485], dtype=uint32)

    In [21]: oidx[ridx == 0]                                                                                                                                                                               
    Out[21]: array([   0,    1,    2, ..., 4483, 4484, 4485], dtype=uint32)

    In [22]: oidx[ridx == 0]                                                                                                                                                                               
    Out[22]: array([   0,    1,    2, ..., 4483, 4484, 4485], dtype=uint32)

    In [23]: len(oidx[ridx == 0])                                                                                                                                                                          
    Out[23]: 4486

    In [24]: len(oidx[ridx != 0])                                                                                                                                                                          
    Out[24]: 7744

    In [25]: pidx[ridx > 0]                                                                                                                                                                                
    Out[25]: array([  0,   1,   2, ..., 861, 862, 863], dtype=uint32)

    In [26]: pidx[ridx > 0].min()                                                                                                                                                                          
    Out[26]: 0

    In [27]: pidx[ridx > 0].max()                                                                                                                                                                          
    Out[27]: 1791

    In [28]: oidx[ridx > 0].max()                                                                                                                                                                          
    Out[28]: 4

    In [29]: oidx[ridx > 0].min()                                                                                                                                                                          
    Out[29]: 0







