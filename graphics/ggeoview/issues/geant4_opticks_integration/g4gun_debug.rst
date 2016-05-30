G4GUN Debug
============

Integration starting point : pure g4gun 
-----------------------------------------

From ggv-g4gun-notes::

   ggv-;ggv-g4gun
         # geant4 particle gun simulation within default DYB geometry, loaded from GDML

   ggv-;ggv-g4gun --load --target 3153
         # visualize the geant4 propagation, with GGeoView
         # see also issues/nopstep_vis_debug.rst  

   ggv-;ggv-g4gun --dbg --load --target 3153 --optixviz

         # attempting to load the CFG4 Geant4/CPU simulated nopstep,photons,records,history
         # needs the optixviz in order to setup OpEngine for indexing the history sequence
         #
         # TODO: revive/re-implement CPU indexer, so can do the lot without GPU (albeit very slowly)


Evt debug with npy-/g4gun.py
-----------------------------

domains
~~~~~~~~~

position/time/wavelength domains::

    simon:cfg4 blyth$ ii
    SQLITE3_DATABASE=/usr/local/env/nuwa/mocknuwa.db
    Python 2.7.11 (default, Dec  5 2015, 23:51:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    IPython profile: g4opticks

    In [1]: run g4gun.py 
    Evt(-1,"G4Gun","G4Gun","G4Gun/G4Gun/-1 : ", seqs="[]")
     fdom :       (3, 1, 4) : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
     idom :       (1, 1, 4) : (metadata) int domain 
       ox :      (96, 4, 4) : (photons) final photon step 
       wl :           (96,) : (photons) wavelength 
     post :         (96, 4) : (photons) final photon step: position, time 
     dirw :         (96, 4) : (photons) final photon step: direction, weight  
     polw :         (96, 4) : (photons) final photon step: polarization, wavelength  
    flags :           (96,) : (photons) final photon step: flags  
       c4 :           (96,) : (photons) final photon step: dtype split uint8 view of ox flags 
       rx :  (96, 10, 2, 4) : (records) photon step records 
       ph :      (96, 1, 2) : (records) photon history flag/material sequence 

    In [2]: evt.fdom
    Out[2]: 
    A(fdomG4Gun,-1,G4Gun)(metadata) 3*float4 domains of position, time, wavelength (used for compression)
    A([[[ -16520.   , -802110.   ,   -7125.   ,    7710.625]],

           [[      0.   ,     200.   ,      50.   ,       0.   ]],

           [[     60.   ,     820.   ,      20.   ,     760.   ]]], dtype=float32)



Previously lacked positional domain (center,extent) causing all positions are at origin::

    In [8]: evt.rpost_(slice(0,5))
    Out[8]: 
    A()sliced
    A([[[  0.   ,   0.   ,   0.   ,   0.012],
            [  0.   ,   0.   ,   0.   ,   8.264],
            [  0.   ,   0.   ,   0.   ,   8.319],
            [  0.   ,   0.   ,   0.   ,  10.566],
            [  0.   ,   0.   ,   0.   ,  10.663]],

           [[  0.   ,   0.   ,   0.   ,   6.738],
            [  0.   ,   0.   ,   0.   ,   6.97 ],
            [  0.   ,   0.   ,   0.   ,   8.319],
            [  0.   ,   0.   ,   0.   ,  10.566],
            [  0.   ,   0.   ,   0.   ,  10.663]],


After fix::

    In [3]: evt.rpost_(slice(0,5))
    Out[3]: 
    A()sliced
    A([[[ -18079.444, -799699.415,   -6603.538,       0.012],
            [ -19508.758, -800298.767,   -6191.734,       8.264],
            [ -19518.171, -800302.767,   -6189.145,       8.319],
            [ -19907.15 , -800465.842,   -6077.134,      10.566],
            [ -19923.857, -800472.901,   -6072.193,      10.663]],

           [[ -18079.444, -799699.415,   -6604.95 ,       6.738],
            [ -18094.034, -799738.477,   -6590.831,       6.97 ],
            [ -19518.171, -800302.767,   -6189.145,       8.319],
            [ -19907.15 , -800465.842,   -6077.134,      10.566],
            [ -19923.857, -800472.901,   -6072.193,      10.663]],

           [[ -18079.444, -799699.415,   -6603.538,       4.895],
            [ -18839.753, -799749.773,   -8635.028,      16.254],
            [ -18845.4  , -799750.008,   -8650.088,      16.34 ],
            [ -18999.533, -799760.362,   -9062.128,      18.647],
            [ -19923.857, -800472.901,   -6072.193,      10.663]],

           ..., 
           [[ -18109.565, -799757.538,   -6583.065,       3.821],
            [ -18120.86 , -799746.714,   -6584.713,       3.906],
            [ -17526.215, -801158.144,   -8181.102,      25.373],
            [ -17410.674, -801565.242,   -8713.624,      28.864],
            [ -17405.732, -801582.655,   -8735.979,      29.011]],

           [[ -18038.029, -799830.957,   -6568.476,       1.508],
            [ -19382.158, -800539.496,   -6052.191,       9.723],
            [ -19390.865, -800544.202,   -6048.896,       9.778],
            [ -19765.489, -800741.633,   -5905.118,      12.067],
            [ -19781.255, -800749.869,   -5899.   ,      12.165]],

           [[ -18058.972, -799810.72 ,   -6608.48 ,      41.279],
            [ -18061.09 , -799810.014,   -6609.186,      41.298],
            [ -19390.865, -800544.202,   -6048.896,       9.778],
            [ -19765.489, -800741.633,   -5905.118,      12.067],
            [ -19781.255, -800749.869,   -5899.   ,      12.165]]])


Opticks Space Domain
-----------------------

When operating triangulated its easy to know the positions of all geometry,
ggv- gets center extent from GMergedMesh::

     532 void App::registerGeometry()
     533 {
     ...
     538     m_mesh0 = m_ggeo->getMergedMesh(0);
     539 
     540     m_ggeo->setComposition(m_composition);
     541 
     542     gfloat4 ce0 = m_mesh0->getCenterExtent(0);  // 0 : all geometry of the mesh, >0 : specific volumes
     543     m_opticks->setSpaceDomain( glm::vec4(ce0.x,ce0.y,ce0.z,ce0.w) );
     544 
     545     if(m_evt)
     546     {
     547        // TODO: migrate npy-/NumpyEvt to opop-/OpEvent so this can happen at more specific level 
     548         m_opticks->dumpDomains("App::registerGeometry copy Opticks domains to m_evt");
     549         m_evt->setSpaceDomain(m_opticks->getSpaceDomain());
     550     }
     551 

::

     760 void GMesh::updateBounds()
     761 {
     762     gbbox   bb = findBBox(m_vertices, m_num_vertices);
     763     gfloat4 ce = bb.center_extent() ;
     764 


G4/GDML Space Domain
------------------------

When running with G4/GDML have entirely analytic geometry. 
Have transforms for all volumes, so can get centers of volumes 
by applying them to (0,0,0,1) but this does not provide
extents ? Differences between multiple volumes would get close. 

To get extents need to dynamically cast solids into specific 
shapes. 

* actually: used capabilities of *G4VSolid* in *cg4-/CSolid*

::

   op --cgdmldetector


Geometry Selection
--------------------

* G4 needs the whole geometry
* Op only needs the part relevant to optical photons, 
  typically run with geometrical volume selection 

But the compression is only applied to optical photon steps.
So should apply equivalent geometrical selection to GDML 



::

    126 op-geometry-query-dyb()
    127 {
    128     case $1 in
    129         DYB)  echo "range:3153:12221"  ;;
    130        IDYB)  echo "range:3158:3160" ;;  # 2 volumes : pvIAV and pvGDS
    131        JDYB)  echo "range:3158:3159" ;;  # 1 volume : pvIAV
    132        KDYB)  echo "range:3159:3160" ;;  # 1 volume : pvGDS
    133        LDYB)  echo "range:3156:3157" ;;  # 1 volume : pvOAV
    134        MDYB)  echo "range:3201:3202,range:3153:3154"  ;;  # 2 volumes : first pmt-hemi-cathode and ADE  
    135     esac
    136     # range:3154:3155  SST  Stainless Steel/IWSWater not a good choice for an envelope, just get BULK_ABSORB without going anywhere
    137 }
    138 
    139 op-geometry-setup-dyb()
    140 {
    141     local geo=${1:-DYB}
    142     export OPTICKS_GEOKEY=DAE_NAME_DYB
    143     export OPTICKS_QUERY=$(op-geometry-query-dyb $geo)


Access with OpticksResource::getQuery 
Argh, query parsing done in assimprap-/AssimpSelection.

* to avoid duplication moved selection into *OpticksQuery* for use from *assimprap-* and *cg4-*


shape
~~~~~

::

    In [1]: run g4gun.py
    Evt(-1,"G4Gun","G4Gun","G4Gun/G4Gun/-1 : ", seqs="[]")
     fdom :       (3, 1, 4) : (metadata) float domain 
     idom :       (1, 1, 4) : (metadata) int domain 
       ox :      (96, 4, 4) : (photons) final photon step 
       wl :           (96,) : (photons) wavelength 
     post :         (96, 4) : (photons) final photon step: position, time 
     dirw :         (96, 4) : (photons) final photon step: direction, weight  
     polw :         (96, 4) : (photons) final photon step: polarization, wavelength  
    flags :           (96,) : (photons) final photon step: flags  
       c4 :           (96,) : (photons) final photon step: dtype split uint8 view of ox flags 
       rx :  (96, 10, 2, 4) : (records) photon step records 
       ph :      (96, 1, 2) : (records) photon history flag/material sequence 


Fishy flags in history
~~~~~~~~~~~~~~~~~~~~~~~~

* unexpected zero flags in history
* all those BT are fishy, or smth wrong with material recording ...

::

    In [2]: print evt.history.table
                            -1:G4Gun 
                      4f        0.365             35       [2 ] G4GUN AB
               4cccccccf        0.333             32       [9 ] G4GUN BT BT BT BT BT BT BT AB
                    4ccf        0.052              5       [4 ] G4GUN BT BT AB
                4cc0cccf        0.052              5       [8 ] G4GUN BT BT BT ?0? BT BT AB
              cccbcccccf        0.052              5       [10] G4GUN BT BT BT BT BT BR BT BT BT
                  4ccccf        0.031              3       [6 ] G4GUN BT BT BT BT AB
                     4cf        0.010              1       [3 ] G4GUN BT AB
                4ccccccf        0.010              1       [8 ] G4GUN BT BT BT BT BT BT AB
               4cc0ccc6f        0.010              1       [9 ] G4GUN SC BT BT BT ?0? BT BT AB
              cccccccccf        0.010              1       [10] G4GUN BT BT BT BT BT BT BT BT BT
              4cc00cc0cf        0.010              1       [10] G4GUN BT ?0? BT BT ?0? ?0? BT BT AB
              4cc0cbb0cf        0.010              1       [10] G4GUN BT ?0? BR BR BT ?0? BT BT AB
              4ccccccc6f        0.010              1       [10] G4GUN SC BT BT BT BT BT BT BT AB
              cbcccccccf        0.010              1       [10] G4GUN BT BT BT BT BT BT BT BR BT
              ccbccccc6f        0.010              1       [10] G4GUN SC BT BT BT BT BT BR BT BT
              ccbccccccf        0.010              1       [10] G4GUN BT BT BT BT BT BT BR BT BT
              4cbccccccf        0.010              1       [10] G4GUN BT BT BT BT BT BT BR BT AB
                                  96         1.00 

    In [3]: print evt.material.table
                            -1:G4Gun 
                      11        0.365             35       [2 ] Gd Gd
               111111111        0.344             33       [9 ] Gd Gd Gd Gd Gd Gd Gd Gd Gd
              1111111111        0.135             13       [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Gd
                11111111        0.062              6       [8 ] Gd Gd Gd Gd Gd Gd Gd Gd
                    1111        0.052              5       [4 ] Gd Gd Gd Gd
                  111111        0.031              3       [6 ] Gd Gd Gd Gd Gd Gd
                     111        0.010              1       [3 ] Gd Gd Gd
                                  96         1.00 



