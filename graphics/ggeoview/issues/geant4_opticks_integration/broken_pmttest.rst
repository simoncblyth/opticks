Broken PMT Test 
=================

FIXED 
--------

Fixed by rejig of OpticksQuery passing to CDetector classes, test detector construction with::

    op --cgdmldetector
    op --ctestdetector


Issue : photon record positional qwns all zero
-------------------------------------------------

::

   ggv-pmt-test --cfg4  

Fail to get a ce::

    2016-May-30 15:44:45.481624]:info: CMaker::makeSolid csg Sphere inner 98 outer 99 startTheta 1.70249 deltaTheta 1.4391 endTheta 3.14159
    G4Material WARNING: duplicate name of material OpaqueVacuum
    [2016-May-30 15:44:45.481903]:info: CTestDetector::kludgePhotoCathode
    [2016-May-30 15:44:45.482394]:info: CTraverser::AncestorTraverse  numSelected 0 bbox NBoundingBox low 340282346638528859811704183484516925440.0000,340282346638528859811704183484516925440.0000,340282346638528859811704183484516925440.0000 high -340282346638528859811704183484516925440.0000,-340282346638528859811704183484516925440.0000,-340282346638528859811704183484516925440.0000 ce 0.0000,0.0000,0.0000,0.0000
    [2016-May-30 15:44:45.482756]:info: CDetector::traverse numMaterials 6 numMaterialsWithoutMPT 0
    [2016-May-30 15:44:45.482901]:info: CG4::configureGenerator TORCH 
    [2016-May-30 15:44:45.483206]:warning: TorchStepNPY::configure skip empty value for key mode


Histories *pmt_test.py* OK, but distrib other than polarization ABC are all zero::

    In [7]: run pmt_test_distrib.py

    ===================================== ====== ====== ========= ====== ===== ===== ===== ======= 
    4/PmtInBox/torch : 107598/107437  :   X      Y      Z         T      A     B     C     R       
    ===================================== ====== ====== ========= ====== ===== ===== ===== ======= 
    [TO] BT SD                            997.68 997.38 107517.50  0.12   0.96  1.04  0.12 2009.67 
    TO [BT] SD                            997.68 997.38 3359.92   428.58  0.96  1.04  0.12 2009.67 
    TO BT [SD]                            988.05 988.49 2687.94   315.81  0.96  1.04  0.12 1991.06 
    ===================================== ====== ====== ========= ====== ===== ===== ===== ======= 


After fix, back to just time being off::

    ===================================== ===== ===== ===== ====== ===== ===== ===== ===== 
    4/PmtInBox/torch : 107598/107437  :   X     Y     Z     T      A     B     C     R     
    ===================================== ===== ===== ===== ====== ===== ===== ===== ===== 
    [TO] BT SD                             0.97  0.94  0.12  0.12   0.96  1.04  0.12  1.11 
    TO [BT] SD                             0.97  0.94  1.08 428.58  0.96  1.04  0.12  1.11 
    TO BT [SD]                             1.01  0.84  1.29 315.81  0.96  1.04  0.12  1.18 
    ===================================== ===== ===== ===== ====== ===== ===== ===== ===== 




