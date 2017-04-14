pmt-src(){      echo ana/pmt/pmt.bash ; }
pmt-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(pmt-src)} ; }
pmt-vi(){       vi $(pmt-source) ; }


pmt-env(){      olocal- ; }
pmt-usage(){ cat << EOU

Analytic PMT Geometry Description
======================================

TODO
-----

* place xml PMT sources into opticksdata where can be generally accessible


FUNCTIONS
-----------

*pmt-analytic*
     runs analytic.py converting detdesc hemi-pmt.xml into parts buffer $IDPATH/GPmt/0/GPmt.npy 
     using more nuanced translation better suited to surface geometry lingo 

Usage example::

    simon:ana blyth$ pmt-analytic --apmtidx=3
    /Users/blyth/opticks/ana/pmt/analytic.py --apmtidx=3

    Aiming to write serialized analytic PMT to below apmtpath
    $OPTICKS_INSTALL_PREFIX/opticksdata/export/DayaBay/GPmt/3/GPmt.npy

    Enter YES to proceed... 




TESTS
------

*pmt-parts*
     runs tree.py converting detdesc hemi-pmt.xml into parts buffer 
     using a direct translation approach, does not save the PMT, used
     just for testing conversion 

*pmt-dd*
     test detdesc parsing 

*pmt-gcsg*
     does nothing other than testing gcsg.py is valid python  


See Also
---------

* opticks/notes/issues/tpmt_broken_by_OpticksCSG_enum_move.rst


Sources
--------

analytic.py
     top level steering for pmt-analytic, using tree.py and dd.py 

tree.py 
     Assembles tree from Nodes using volume path digest trick

     Buf(np.ndarray)
     Node
     Tree

gcsg.py
     serialization of CSG tree, so far not used GPU side, 
     (the part buffer representation is used GPU side)

     CSG serializarion is however used via ggeo/GCSG for 
     the creation the Geant4 test geometry, including the PMT

     NB GCSG was my initial take on CSG that never worked on GPU, 
     the new way npy/NCSG was designed for GPU using a binary tree
     serialization to get over to GPU side 
     
     cfg4/CMaker
     cfg4/CPropLib
     cfg4/CTestDetector


dd.py 
     detdesc XML parsing using lxml, and Dayabay PMT centric boolean partitioning 
     into single basis shape parts

     Parts(list)
     Uncoincide
     Att
     Elem
     Logvol(Elem)
     Physvol(Elem)
     Union(Elem)
     Intersection(Elem)
     Parameter(Elem)
     Primitive(Elem)  
     Sphere(Primitive)
     Tubs(Primitive)
     PosXYZ(Elem)
     Context
     Dddb(Elem)

plot.py 
     PMT basis shape and also mesh 2d plots,  





Checking the opticksdata/export/DayaBay/GPmt serializations
-------------------------------------------------------------

Used::

   pmt-analytic 


::

    simon:issues blyth$ cd /usr/local/opticks/opticksdata/export/DayaBay/GPmt
    simon:GPmt blyth$ l
    total 0
    drwxr-xr-x   8 blyth  staff  272 Apr 10 15:56 2
    drwxr-xr-x   8 blyth  staff  272 Mar 16 13:15 1
    drwxr-xr-x  12 blyth  staff  408 Jul  5  2016 0
    simon:GPmt blyth$ diff -r 1 2 
    simon:GPmt blyth$ diff -r 0 1
    Binary files 0/GPmt.npy and 1/GPmt.npy differ
    Only in 0: GPmt.txt
    Only in 0: GPmt_check.npy
    Only in 0: GPmt_check.txt
    Only in 0: GPmt_csg.txt
    simon:GPmt blyth$ 



Serialization
----------------

::

    simon:PMT blyth$ l /tmp/blyth/opticks/GPmt/0/
    total 48
    -rw-r--r--  1 blyth  wheel   848 Mar 15 16:35 GPmt.npy
    -rw-r--r--  1 blyth  wheel   289 Mar 15 16:35 GPmt_boundaries.txt
    -rw-r--r--  1 blyth  wheel  1168 Mar 15 16:35 GPmt_csg.npy
    -rw-r--r--  1 blyth  wheel    74 Mar 15 16:35 GPmt_lvnames.txt
    -rw-r--r--  1 blyth  wheel    47 Mar 15 16:35 GPmt_materials.txt
    -rw-r--r--  1 blyth  wheel    74 Mar 15 16:35 GPmt_pvnames.txt

::

    delta:~ blyth$ cd /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/
    delta:0 blyth$ l
    total 80

    -rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt_check.npy
    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_check.txt


    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt.txt             # renamed to _boundaries ?
    -rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_csg.txt         # renamed to _materials ? 

    -rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt.npy
    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_boundaries.txt
    -rw-r--r--  1 blyth  staff  1168 Jul  5  2016 GPmt_csg.npy
    -rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_lvnames.txt
    -rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_materials.txt
    -rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_pvnames.txt

    delta:0 blyth$ cat GPmt_csg.txt  # where is this one written ?  appears to be former name for GPmt_materials.txt
    Pyrex
    Vacuum
    Bialkali
    OpaqueVacuum
    OpaqueVacuum

    delta:0 blyth$ wc -l *.txt    # hmm seems wc actually counts newlines, so add one to all the below
          11 GPmt.txt
          11 GPmt_boundaries.txt
          11 GPmt_check.txt
           4 GPmt_csg.txt
           4 GPmt_lvnames.txt
           4 GPmt_materials.txt
           4 GPmt_pvnames.txt
          49 total

    delta:0 blyth$ head -1 *.npy
    ==> GPmt.npy <==
    ?NUMPYF{'descr': '<f4', 'fortran_order': False, 'shape': (12, 4, 4), }      

    ==> GPmt_check.npy <==
    ?NUMPYF{'descr': '<f4', 'fortran_order': False, 'shape': (12, 4, 4), }      

    ==> GPmt_csg.npy <==
    ?NUMPYF{'descr': '<f4', 'fortran_order': False, 'shape': (17, 4, 4), }      
    delta:0 blyth$ 



Comparing existing serializations
--------------------------------------

All look effectively the same::

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





History
--------

The *pmt* directory hails from **env/nuwa/detdesc/pmt**


Usage
------

To visualize analytic PMT in a box, and test ggeo- optixrap- loading::

    ggv-pmt () 
    { 
        type $FUNCNAME;
        ggv --tracer --test --eye 0.5,0.5,0.0
    }


TODO debug why below is failing::

    ggv --restrictmesh 1 --analyticmesh 1  
          # misses a GPmt associated to the MM

    ggv --test --eye 0.5,0.5,0.0 --animtimemax 10


High Level CSG Persisting for G4 geometry
-------------------------------------------

* how to represent a single node (of the 5 separate nodes) ?

  * primitives and operations

* tree of nodes  


PMT Modelling
-----------------------

Detdesc::

    lvPmtHemi                 (Pyrex)  union of 3 sphere-intersection and tubs
        pvPmtHemiVacuum

    lvPmtHemiVacuum          (Vacuum) union of 3 sphere-intersection and tubs 
        pvPmtHemiCathode
        pvPmtHemiBottom
        pvPmtHemiDynode

        [Vacuum radii names match the Pyrex with "vac" suffix]    

            PmtHemiBellyROC : 102.000000 
         PmtHemiBellyROCvac : 99.000000 

             PmtHemiFaceROC : 131.000000 
          PmtHemiFaceROCvac : 128.000000 

      PmtHemiGlassThickness : 3.000000 
    PmtHemiCathodeThickness : 0.050000 



    lvPmtHemiCathode        (Bialkali) union of 2 partial spherical shells

          outerRadius="PmtHemiFaceROCvac"
          innerRadius="PmtHemiFaceROCvac-PmtHemiCathodeThickness"

          outerRadius="PmtHemiBellyROCvac"
          innerRadius="PmtHemiBellyROCvac-PmtHemiCathodeThickness"      

      **OUTER SURFACE OF CATHODE COINCIDES WITH VACUUM/PYREX INTERFACE* 
        

    lvPmtHemiBottom        (OpaqueVacuum) partial spherical shell

          outerRadius="PmtHemiBellyROCvac"
          innerRadius="PmtHemiBellyROCvac-1*mm"

      **OUTER SURFACE OF BOTTOM COINCIDES WITH VACUUM/PYREX INTERFACE* 
           



    lvPmtHemiDynode        (OpaqueVacuum) tubs

          outerRadius="PmtHemiDynodeRadius" 


            PmtHemiDynodeRadius : 27.500000 



 PmtHemiBellyCathodeAngleDelta : 26.735889 
  PmtHemiBellyCathodeAngleStart : 55.718631 
           PmtHemiBellyIntAngle : 82.454520 
                PmtHemiBellyOff : 13.000000 
         PmtHemiCathodeDiameter : 190.000000 
        PmtHemiFaceCathodeAngle : 40.504998 
                 PmtHemiFaceOff : 56.000000 

              PmtHemiFaceTopOff : 55.046512 
         PmtHemiGlassBaseLength : 169.000000 
         PmtHemiGlassBaseRadius : 42.250000 


Partitioned Boundary Translation
---------------------------------

Cathode Z region, 3 boundary spherical parts::

    MineralOil///Pyrex
    Pyrex/lvPmtHemiCathodeSensorSurface//Bialkali
    Bialkali///Vacuum 

    * different from literal translation:

      MineralOil///Pyrex
      Pyrex///Vacuum                                     <-- coincident
      Vacuum/lvPmtHemiCathodeSensorSurface//Bialkali     <-- coincident
      Bialkali///Vacuum 

Bottom Z region, 3 boundary spherical parts::

    MineralOil///Pyrex
    Pyrex///OpaqueVacuum  
    OpaqueVacuum///Vacuum 

    * different from literal translation (with zero size slice Vacuum)

      MineralOil///Pyrex
      Pyrex///Vacuum                <--- coincident
      Vacuum///OpaqueVacuum         <--- coincident
      OpaqueVacuum///Vacuum 

    * Bottom OpaqueVacuum is 1mm thick, but Cathode is 0.05mm stuck to inside of Pyrex
      so 0.95mm of protuding OpaqueVacuum : what will happen to photons hitting that 
      protuberance ...  
      BUT the Cathode in "std" model absorbs/detects all photons that hit it, so 
      probably do not need to worry much about loose edges inside ?


Dynode Z region, 3 boundary tubs parts::

    MineralOil///Pyrex
    Pyrex///Vacuum
    Vacuum///OpaqueVacuum
 
    * dynode tubs overlaps the bottom spherical shell



What about joining up the Z regions ? 

* Does the BBox approach auto-close open ends ? need to shoot lots of photons and see..

* MineralOil///Pyrex is no problem, as Z splits chosen for contiguity 
 


Implementation
----------------

* Surface model identities must diverge from Triangulated due to translation differences
  so need to label the parts with these boundaries  

Original direct translation::

    Part Sphere        Pyrex    pmt-hemi-bot-glass_part_zleft    [0, 0, 69.0] r: 102.0 sz:  0.0 bb:BBox      [-101.17 -101.17  -23.84]      [ 101.17  101.17   56.  ] xyz [  0.     0.    16.08]
    Part Sphere        Pyrex  pmt-hemi-top-glass_part_zmiddle    [0, 0, 43.0] r: 102.0 sz:  0.0 bb:BBox      [-101.17 -101.17   56.  ]      [ 101.17  101.17  100.07] xyz [  0.     0.    78.03]
    Part Sphere        Pyrex  pmt-hemi-face-glass_part_zright       [0, 0, 0] r: 131.0 sz:  0.0 bb:BBox      [ -84.54  -84.54  100.07]      [  84.54   84.54  131.  ] xyz [   0.      0.    115.53]
    Part   Tubs        Pyrex               pmt-hemi-base_part   [0, 0, -84.5] r: 42.25 sz:169.0 bb:BBox      [ -42.25  -42.25 -169.  ]         [ 42.25  42.25 -23.84] xyz [  0.     0.   -96.42]

    Part Sphere       Vacuum    pmt-hemi-face-vac_part_zright       [0, 0, 0] r: 128.0 sz:  0.0 bb:BBox         [-82.29 -82.29  98.05]      [  82.29   82.29  128.  ] xyz [   0.      0.    113.02]
    Part Sphere       Vacuum    pmt-hemi-top-vac_part_zmiddle    [0, 0, 43.0] r:  99.0 sz:  0.0 bb:BBox         [-98.14 -98.14  56.  ]         [ 98.14  98.14  98.05] xyz [  0.     0.    77.02]
    Part Sphere       Vacuum      pmt-hemi-bot-vac_part_zleft    [0, 0, 69.0] r:  99.0 sz:  0.0 bb:BBox         [-98.14 -98.14 -21.89]         [ 98.14  98.14  56.  ] xyz [  0.     0.    17.06]
    Part   Tubs       Vacuum           pmt-hemi-base-vac_part   [0, 0, -81.5] r: 39.25 sz:166.0 bb:BBox      [ -39.25  -39.25 -164.5 ]         [ 39.25  39.25 -21.89] xyz [  0.     0.   -93.19]

    Part Sphere     Bialkali       pmt-hemi-cathode-face_part       [0, 0, 0] r: 128.0 sz:  0.0 bb:BBox         [-82.29 -82.29  98.05]      [  82.29   82.29  128.  ] xyz [   0.      0.    113.02]
    Part Sphere     Bialkali       pmt-hemi-cathode-face_part       [0, 0, 0] r:127.95 sz:  0.0 bb:BBox         [-82.25 -82.25  98.01]      [  82.25   82.25  127.95] xyz [   0.      0.    112.98]
    Part Sphere     Bialkali      pmt-hemi-cathode-belly_part    [0, 0, 43.0] r:  99.0 sz:  0.0 bb:BBox         [-98.14 -98.14  56.  ]         [ 98.14  98.14  98.05] xyz [  0.     0.    77.02]
    Part Sphere     Bialkali      pmt-hemi-cathode-belly_part    [0, 0, 43.0] r: 98.95 sz:  0.0 bb:BBox         [-98.09 -98.09  55.99]         [ 98.09  98.09  98.01] xyz [  0.   0.  77.]

    Part Sphere OpaqueVacuum                pmt-hemi-bot_part    [0, 0, 69.0] r:  99.0 sz:  0.0 bb:BBox         [-98.14 -98.14 -30.  ]         [ 98.14  98.14  56.  ] xyz [  0.   0.  13.]
    Part Sphere OpaqueVacuum                pmt-hemi-bot_part    [0, 0, 69.0] r:  98.0 sz:  0.0 bb:BBox         [-97.15 -97.15 -29.  ]         [ 97.15  97.15  56.13] xyz [  0.     0.    13.57]
    Part   Tubs OpaqueVacuum             pmt-hemi-dynode_part   [0, 0, -81.5] r:  27.5 sz:166.0 bb:BBox         [ -27.5  -27.5 -164.5]            [ 27.5  27.5   1.5] xyz [  0.    0.  -81.5]


With coincident surface removal and boundary name rejig and persisting as bndspec list GPmt.txt::

    Part Sphere        Pyrex    pmt-hemi-bot-glass_part_zleft    [0, 0, 69.0] r: 102.0 sz:  0.0 BB      [-101.17 -101.17  -23.84]      [ 101.17  101.17   56.  ] z  16.08 MineralOil///Pyrex
    Part Sphere        Pyrex  pmt-hemi-top-glass_part_zmiddle    [0, 0, 43.0] r: 102.0 sz:  0.0 BB      [-101.17 -101.17   56.  ]      [ 101.17  101.17  100.07] z  78.03 MineralOil///Pyrex
    Part Sphere        Pyrex  pmt-hemi-face-glass_part_zright       [0, 0, 0] r: 131.0 sz:  0.0 BB      [ -84.54  -84.54  100.07]      [  84.54   84.54  131.  ] z 115.53 MineralOil///Pyrex
    Part   Tubs        Pyrex               pmt-hemi-base_part   [0, 0, -84.5] r: 42.25 sz:169.0 BB      [ -42.25  -42.25 -169.  ]         [ 42.25  42.25 -23.84] z -96.42 MineralOil///Pyrex
    Part Sphere       Vacuum      pmt-hemi-bot-vac_part_zleft    [0, 0, 69.0] r:  99.0 sz:  0.0 BB         [-98.14 -98.14 -21.89]         [ 98.14  98.14  56.  ] z  17.06 Pyrex///OpaqueVacuum
    Part Sphere       Vacuum    pmt-hemi-top-vac_part_zmiddle    [0, 0, 43.0] r:  99.0 sz:  0.0 BB         [-98.14 -98.14  56.  ]         [ 98.14  98.14  98.05] z  77.02 Pyrex/lvPmtHemiCathodeSensorSurface//Bialkali
    Part Sphere       Vacuum    pmt-hemi-face-vac_part_zright       [0, 0, 0] r: 128.0 sz:  0.0 BB         [-82.29 -82.29  98.05]      [  82.29   82.29  128.  ] z 113.02 Pyrex/lvPmtHemiCathodeSensorSurface//Bialkali
    Part   Tubs       Vacuum           pmt-hemi-base-vac_part   [0, 0, -81.5] r: 39.25 sz:166.0 BB      [ -39.25  -39.25 -164.5 ]         [ 39.25  39.25 -21.89] z -93.19 Pyrex///Vacuum
    Part Sphere     Bialkali       pmt-hemi-cathode-face_part       [0, 0, 0] r:127.95 sz:  0.0 BB         [-82.25 -82.25  98.01]      [  82.25   82.25  127.95] z 112.98 Bialkali///Vacuum
    Part Sphere     Bialkali      pmt-hemi-cathode-belly_part    [0, 0, 43.0] r: 98.95 sz:  0.0 BB         [-98.09 -98.09  55.99]         [ 98.09  98.09  98.01] z  77.00 Bialkali///Vacuum
    Part Sphere OpaqueVacuum                pmt-hemi-bot_part    [0, 0, 69.0] r:  98.0 sz:  0.0 BB         [-97.15 -97.15 -29.  ]         [ 97.15  97.15  56.13] z  13.57 OpaqueVacuum///Vacuum
    Part   Tubs OpaqueVacuum             pmt-hemi-dynode_part   [0, 0, -81.5] r:  27.5 sz:166.0 BB         [ -27.5  -27.5 -164.5]            [ 27.5  27.5   1.5] z -81.50 Vacuum///OpaqueVacuum
    [tree.py +214                 save ] saving to $IDPATH/GPmt/0/GPmt.npy shape (12, 4, 4) 
    [tree.py +217                 save ] saving boundaries to /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPmt/0/GPmt.txt 


EOU
}

pmt-cd(){  cd $(pmt-dir); }
pmt-ecd(){ cd $(pmt-edir) ; }

pmt-xml(){ vi $(pmt-dir)/hemi-pmt.xml ; }
pmt-dir(){ echo $(local-base)/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT ; }

pmt-edir(){ echo $(opticks-home)/ana/pmt ; }
pmt-export(){  
    echo -n 
    # PMT_DIR env setup moved into opticks_main in ana.base
}

pmt-i(){
   pmt-ecd
   i
}

## TODO: consolidate, too many entry points 
pmt-run(){      python $(pmt-edir)/${1:-pmt}.py  ; }
pmt-dd(){       python $(pmt-edir)/dd.py  ; }
pmt-parts(){    python $(pmt-edir)/tree.py $*  ; }
pmt-analytic(){ python $(pmt-edir)/analytic.py $*  ; }
pmt-gcsg(){     python $(pmt-edir)/gcsg.py $*  ; }



