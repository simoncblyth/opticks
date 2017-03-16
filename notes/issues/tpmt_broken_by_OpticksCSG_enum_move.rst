tpmt broken by OpticksCSG enum move
======================================

* shape/operator enum unification to use sysrap/OpticksCSG.{h,py} is incomplete
* tpmt broken due to mis-interpretation of part buffer

overview
----------

* old PMT serialization needs to be rebuilt with new unified enum   
* rebuilt analytic PMT and stored into opticksdata with non-default apmtidx slot 1 (not committed)


symptom 2 : CPU/G4 cfg4/CTestDetector misunderstanding primordial CSG buffer ?
-----------------------------------------------------------------------------------

* actually the PmtInBox code appears to be unaware of GCSG 

::

    tpmt-- --okg4

    2017-03-16 13:51:10.046 INFO  [889146] [OpticksGen::targetGenstep@125] OpticksGen::targetGenstep setting frame 1 1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 0.0000,0.0000,0.0000,1.0000
    2017-03-16 13:51:10.047 FATAL [889146] [GenstepNPY::setPolarization@212] GenstepNPY::setPolarization pol 0.0000,0.0000,0.0000,0.0000 npol nan,nan,nan,nan m_polw nan,nan,nan,380.0000
    2017-03-16 13:51:10.047 INFO  [889146] [SLog::operator@15] OpticksHub::OpticksHub DONE

    *************************************************************
     Geant4 version Name: geant4-10-02-patch-01    (26-February-2016)
                          Copyright : Geant4 Collaboration
                          Reference : NIM A 506 (2003), 250-303
                                WWW : http://cern.ch/geant4
    *************************************************************

    2017-03-16 13:51:10.122 FATAL [889146] [CGeometry::init@59] CGeometry::init G4 simple test geometry 
    2017-03-16 13:51:10.122 INFO  [889146] [GGeo::createSurLib@656] deferred creation of GSurLib 
    2017-03-16 13:51:10.122 INFO  [889146] [GSurLib::collectSur@79]  nsur 48
    2017-03-16 13:51:10.122 INFO  [889146] [CPropLib::init@68] CPropLib::init
    2017-03-16 13:51:10.122 INFO  [889146] [CPropLib::initCheckConstants@120] CPropLib::initCheckConstants mm 1 MeV 1 nanosecond 1 ns 1 nm 1e-06 GC::nanometer 1e-06 h_Planck 4.13567e-12 GC::h_Planck 4.13567e-12 c_light 299.792 GC::c_light 299.792 dscale 0.00123984
    2017-03-16 13:51:10.122 INFO  [889146] [*CTestDetector::makeDetector@118] CTestDetector::makeDetector PmtInBox 1 BoxInBox 0 numSolids (from mesh0) 7 numSolids (from config) 1
    Assertion failed: (numSolids == numSolidsConfig), function makeDetector, file /Users/blyth/opticks/cfg4/CTestDetector.cc, line 127.
    /Users/blyth/opticks/bin/op.sh: line 580: 41465 Abort trap: 6           /usr/local/opticks/lib/OKG4Test --anakey tpmt --save --test --testconfig mode=PmtInBox_pmtpath=/usr/local/opticks/opticksdata/export/dpib/GMergedMesh/0_control=1,0,0,0_analytic=1_apmtidx=1_node=box_parameters=0,0,0,300_boundary=Rock/NONE/perfectAbsorbSurface/MineralOil --torch --torchconfig type=disc_photons=500000_wavelength=380_frame=1_source=0,0,300_target=0,0,0_radius=100_zenithazimuth=0,1,0,1_material=Vacuum_mode=_polarization= --cat PmtInBox --tag 10 --timemax 10 --animtimemax 10 --eye 0.0,-0.5,0.0 --geocenter --okg4
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:opticks blyth$ 


    2017-03-16 14:17:21.209 INFO  [901864] [CPropLib::initCheckConstants@120] CPropLib::initCheckConstants mm 1 MeV 1 nanosecond 1 ns 1 nm 1e-06 GC::nanometer 1e-06 h_Planck 4.13567e-12 GC::h_Planck 4.13567e-12 c_light 299.792 GC::c_light 299.792 dscale 0.00123984
    2017-03-16 14:17:21.209 INFO  [901864] [*CTestDetector::makeDetector@118] CTestDetector::makeDetector PmtInBox 1 BoxInBox 0 numSolidsMesh 7 numSolidsConfig 1
    2017-03-16 14:17:21.209 INFO  [901864] [GMergedMesh::dumpSolids@617] CTestDetector::makeDetector (solid count inconsistent)
        0 ce             gfloat4      0.000      0.000      0.000    300.000  bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000  ni(         0,         0,         0,4294967295) id(         0,         5,         0,         0)
        1 ce             gfloat4      0.000      0.000    -18.997    149.997  bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000  ni(       720,       362,         1,         0) id(         1,         4,         1,         0)
        2 ce             gfloat4      0.000      0.000    -18.247    146.247  bb bb min    -97.288    -97.288   -164.495  max     97.288     97.288    128.000  ni(       720,       362,         2,         1) id(         2,         3,         2,         0)
        3 ce             gfloat4      0.005      0.004     91.998     98.143  bb bb min    -98.138    -98.139     55.996  max     98.148     98.147    128.000  ni(       960,       482,         3,         2) id(         3,         0,         3,         0)
        4 ce             gfloat4      0.000      0.000     13.066     98.143  bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131  ni(       576,       288,         4,         2) id(         4,         1,         4,         0)
        5 ce             gfloat4      0.000      0.000    -81.500     83.000  bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500  ni(        96,        50,         5,         2) id(         5,         2,         4,         0)
        6 ce             gfloat4      0.000      0.000      0.000    300.000  bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000  ni(        12,        24,         0,4294967295) id(         0,      1000,         0,         0)
    Assertion failed: (numSolidsMesh == numSolidsConfig), function makeDetector, file /Users/blyth/opticks/cfg4/CTestDetector.cc, line 133.


looks like okg4 not updated since primordial GCSG 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Approach 

* make connection between the analytic GCSG volumes that CTestDetector::makePMT 
  is going to use and the triangulated GMergedMesh solid count, 
  then can update the assert

* avoid duplicity regards the analytic PMT and honour the apmtidx version, by 
  eliminating CPropLib::getPmtCSG

::

    simon:opticks blyth$ opticks-find getPmtCSG
    ./cfg4/CPropLib.cc:GCSG* CPropLib::getPmtCSG(NSlice* slice)
    ./cfg4/CPropLib.cc:        LOG(error) << "CPropLib::getPmtCSG failed to load PMT" ;
    ./cfg4/CPropLib.cc:        LOG(error) << "CPropLib::getPmtCSG failed to getCSG from GPmt" ;
    ./cfg4/CTestDetector.cc:    GCSG* csg = m_mlib->getPmtCSG(slice);
    ./cfg4/CPropLib.hh:       GCSG*       getPmtCSG(NSlice* slice);


    162 GCSG* CPropLib::getPmtCSG(NSlice* slice)
    163 {
    164    // hmm this is probably already loaded ???
    165    
    166     GPmt* pmt = GPmt::load( m_ok, m_bndlib, 0, slice );    // pmtIndex:0
    167     
    168     if(pmt == NULL)
    169     {
    170         LOG(error) << "CPropLib::getPmtCSG failed to load PMT" ;
    171         return NULL ; 
    172     }   
    173     
    174     GCSG* csg = pmt->getCSG();
    175     
    176     if(csg == NULL)
    177     {
    178         LOG(error) << "CPropLib::getPmtCSG failed to getCSG from GPmt" ;
    179         return NULL ; 
    180     }   
    181     return csg ;
    182 }   





FIXED : symptom 1, GPU side mis-interpreting parts buffer after enum change
-----------------------------------------------------------------------------

::

    tpmt--   

    2017-03-15 20:48:44.712 INFO  [829428] [OContext::close@219] OContext::close numEntryPoint 2
    ##hemi-pmt.cu:bounds primIdx 0 is_partlist:0 min  -101.1682  -101.1682   -23.8382 max   101.1682   101.1682    56.0000 
    ##hemi-pmt.cu:bounds primIdx 1 is_partlist:0 min   -98.1428   -98.1428    56.0000 max    98.1428    98.1428    98.0465 
    ##hemi-pmt.cu:bounds primIdx 2 is_partlist:0 min   -98.0932   -98.0932    55.9934 max    98.0932    98.0932    98.0128 
    ##hemi-pmt.cu:bounds primIdx 3 is_partlist:0 min   -27.5000   -27.5000  -164.5000 max    27.5000    27.5000     1.5000 
    ##hemi-pmt.cu:bounds primIdx 4 is_partlist:0 min  -300.0100  -300.0100  -300.0100 max   300.0100   300.0100   300.0100 
    2017-03-15 20:48:45.342 INFO  [829428] [OPropagator::prelaunch@149] 1 : (0;500000,1) prelaunch_times vali,comp,prel,lnch  0.0000 0.2694 0.2364 0.0000
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit


review of analytic PMT serialization
--------------------------------------

* ana/pmt/analytic.py 

Recreate the analytic PMT from detdecs parse with

::

   pmt-analytic-tmp   # writing to $TMP/GPmt/0/GPmt.npy
   pmt-analytic       # writing to $IDPATH/GPmt/0/GPmt.npy

Actual one in use is from opticksdata repo $OPTICKS_DATA/export/DayaBay/GPmt/0/  


Comparing existing serializations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three look effectively the same, with no influence from new enum so far::

    simon:pmt blyth$ l /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPmt/0/
    total 48
    -rw-r--r--  1 blyth  staff   848 Mar 15 16:27 GPmt.npy
    -rw-r--r--  1 blyth  staff   289 Mar 15 16:27 GPmt_boundaries.txt
    -rw-r--r--  1 blyth  staff  1168 Mar 15 16:27 GPmt_csg.npy
    -rw-r--r--  1 blyth  staff    74 Mar 15 16:27 GPmt_lvnames.txt
    -rw-r--r--  1 blyth  staff    47 Mar 15 16:27 GPmt_materials.txt
    -rw-r--r--  1 blyth  staff    74 Mar 15 16:27 GPmt_pvnames.txt
    simon:pmt blyth$ 
    simon:pmt blyth$ 
    simon:pmt blyth$ l $TMP/GPmt/0/
    total 48
    -rw-r--r--  1 blyth  wheel   848 Mar 15 17:31 GPmt.npy
    -rw-r--r--  1 blyth  wheel   289 Mar 15 17:31 GPmt_boundaries.txt
    -rw-r--r--  1 blyth  wheel  1168 Mar 15 17:31 GPmt_csg.npy
    -rw-r--r--  1 blyth  wheel    74 Mar 15 17:31 GPmt_lvnames.txt
    -rw-r--r--  1 blyth  wheel    47 Mar 15 17:31 GPmt_materials.txt
    -rw-r--r--  1 blyth  wheel    74 Mar 15 17:31 GPmt_pvnames.txt
    simon:pmt blyth$ diff -r --brief $IDPATH/GPmt/0 $TMP/GPmt/0
    simon:pmt blyth$ 
    simon:pmt blyth$ 
    simon:pmt blyth$ l /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/
    total 80
    -rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt.npy
    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt.txt
    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_boundaries.txt
    -rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt_check.npy
    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_check.txt
    -rw-r--r--  1 blyth  staff  1168 Jul  5  2016 GPmt_csg.npy
    -rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_csg.txt
    -rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_lvnames.txt
    -rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_materials.txt
    -rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_pvnames.txt

    simon:pmt blyth$ echo $OPTICKS_DATA
    /usr/local/opticks/opticksdata
    simon:pmt blyth$ 
    simon:pmt blyth$ diff -r --brief $OPTICKS_DATA/export/DayaBay/GPmt/0/ $TMP/GPmt/0/
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt.txt
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt_check.npy
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt_check.txt
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt_csg.txt





