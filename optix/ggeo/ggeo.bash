# === func-gen- : optix/ggeo/ggeo fgp optix/ggeo/ggeo.bash fgn ggeo fgh optix/ggeo
ggeo-rel(){      echo optix/ggeo ; }
ggeo-src(){      echo optix/ggeo/ggeo.bash ; }
ggeo-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ggeo-src)} ; }
ggeo-vi(){       vi $(ggeo-source) ; }
ggeo-usage(){ cat << EOU

GGEO : Intermediary Geometry Model
=====================================

Unencumbered geometry and material/surface property
model, intended for:

* investigation of how to represent Geometry within OptiX


TODO
-----

* equivalent of g4daenode.py:add_sensitive_surfaces
* idmap handling, for PMTid into OptiX

* migration of GLoader into GGeo has brought in an AssimpWrap dependency
  thats kinda unhealthy : it builds OK but a bit circular
 
  * lookinto arranging the GLoader to avoid that ? 


REVIEW
-------

* rejig relationship between GBoundaryLib and GBoundary
  
  * lib doing too much, substance doing too little

  * move standardization from the lib to the substance
    so that lib keys are standard digests, this will 
    allow the standard lib to be reconstructed from the 
    wavelengthBuffer and will offer simple digest matching
    to the metadata 

  * can have a separate non-standardized lib intstance 
    as a container for all properties 


Matplotlib based gui for looking at material properties ?
-----------------------------------------------------------

wxwidgets
~~~~~~~~~~~

* :google:`python matplotlib gui`
* https://pypi.python.org/pypi/plotexplorer_gui
* http://eli.thegreenplace.net/2008/08/01/matplotlib-with-wxpython-guis/

* http://wiki.scipy.org/Cookbook/Matplotlib/EmbeddingInWx

* http://agni.phys.iit.edu/~kmcivor/wxmpl/
* http://agni.phys.iit.edu/~kmcivor/wxmpl/tutorial/

qt
~~~

* http://pyqtgraph.org


Material Indices mismatch
--------------------------

::

    In [1]: o = np.load("optical.npy")

    In [14]: oo = o.reshape(-1,6,4).view(np.uint32)

    In [21]: oo[3:10]
    Out[21]: 
    array([[[  4,   0,   0,   0],
            [  3,   0,   0,   0],
            [  0,   0,   0,   0],
            [  1,   0,   3, 100],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0]],

           [[  5,   0,   0,   0],
            [  3,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0]],
 
            ...

    In [19]: np.unique(oo[:, 0, 0])
    Out[19]: 
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25], dtype=uint32)

    In [20]: np.unique(oo[:, 1, 0])
    Out[20]: array([ 1,  2,  3,  5,  7,  8,  9, 10, 11, 12, 13, 14, 16, 20, 21, 23], dtype=uint32)



The materials order in .opticks is shuffled to put important materials first, all the material .json in .opticks
are the same::

    simon:.opticks blyth$ l
    total 40
    -rw-r--r--  1 blyth  staff   548 Jul  8 13:00 GMaterialIndexLocal.json
    -rw-r--r--  1 blyth  staff   548 Jul  8 13:00 GMaterialIndexSource.json
    -rw-r--r--  1 blyth  staff   548 Jul  5 16:33 MaterialsLocal.json
    -rw-r--r--  1 blyth  staff   548 Jul  5 16:33 MaterialsSource.json
    -rw-r--r--  1 blyth  staff  3514 Jun 21 10:39 GColors.json
    simon:.opticks blyth$ diff GMaterialIndexLocal.json GMaterialIndexSource.json
    simon:.opticks blyth$ vi MaterialsLocal.json
    simon:.opticks blyth$ diff MaterialsLocal.json GMaterialIndexSource.json
    simon:.opticks blyth$ diff MaterialsLocal.json MaterialsSource.json


::

    In [6]: m = json.load(file(os.path.expanduser("~/.opticks/GMaterialIndexLocal.json")))

    In [12]: im = dict(zip(map(int,m.values()),map(str,m.keys())))

    In [13]: im
    Out[13]: 
    {1: 'GdDopedLS',
     2: 'LiquidScintillator',
     3: 'Acrylic',
     4: 'MineralOil',
     5: 'Bialkali',
     6: 'IwsWater',
     7: 'Water',
     8: 'DeadWater',
     9: 'OwsWater',
     10: 'ESR',
     11: 'UnstStainlessSteel',
     12: 'StainlessSteel',


    # material pairs for each boundary from optical.npy translated to names with .opticks json

    In [12]: for i,b in enumerate(oo):print "%2d %2d : %25s (%2d) %25s (%2d) " % ( i,i+1,im[b[0,0]],b[0,0], im[b[1,0]],b[1,0] )
     0  1 :                 GdDopedLS ( 1)                 GdDopedLS ( 1) 
     1  2 :        LiquidScintillator ( 2)                 GdDopedLS ( 1) 
     2  3 :                   Acrylic ( 3)        LiquidScintillator ( 2) 
     3  4 :                MineralOil ( 4)                   Acrylic ( 3) 
     4  5 :                  Bialkali ( 5)                   Acrylic ( 3) 
     5  6 :                  IwsWater ( 6)                  Bialkali ( 5) 
     6  7 :                   Acrylic ( 3)                   Acrylic ( 3) 
     7  8 :                     Water ( 7)        LiquidScintillator ( 2) 
     8  9 :                 DeadWater ( 8)                     Water ( 7) 
     9 10 :                  OwsWater ( 9)                 DeadWater ( 8) 
    10 11 :                 DeadWater ( 8)                  OwsWater ( 9) 
    11 12 :                       ESR (10)                       ESR (10) 
    12 13 :        UnstStainlessSteel (11)                       ESR (10) 
    13 14 :            StainlessSteel (12)        UnstStainlessSteel (11) 
    14 15 :                    Vacuum (13)            StainlessSteel (12) 
    15 16 :                     Pyrex (14)                    Vacuum (13) 
    16 17 :                    Vacuum (13)                     Pyrex (14) 
    17 18 :                       Air (15)                    Vacuum (13) 
    18 19 :                       Air (15)                     Pyrex (14) 
    19 20 :                      Rock (16)            StainlessSteel (12) 
    20 21 :                 GdDopedLS ( 1)                      Rock (16) 
    21 22 :                       PPE (17)                 GdDopedLS ( 1) 
    22 23 :                 Aluminium (18)            StainlessSteel (12) 
    23 24 :                 GdDopedLS ( 1)            StainlessSteel (12) 
    24 25 :                    Vacuum (13)            StainlessSteel (12) 
    25 26 :                   Acrylic ( 3)                    Vacuum (13) 
    26 27 :     ADTableStainlessSteel (19)                   Acrylic ( 3) 
    27 28 :                     Pyrex (14)            StainlessSteel (12) 
    28 29 :                    Vacuum (13)                 GdDopedLS ( 1) 
    29 30 :                       Air (15)        UnstStainlessSteel (11) 

After moving material index customization prior to buffer creation in GLoader
and recreating the geocache the indices in the optical buffer have been shuffled differently::

    simon:ggeo blyth$ ./optical_buffer.py 
     0  1 :                    Vacuum (13)                    Vacuum (13) 
     1  2 :                      Rock (16)                    Vacuum (13) 
     2  3 :                       Air (15)                      Rock (16) 
     3  4 :                       PPE (17)                       Air (15) 
     4  5 :                 Aluminium (18)                       Air (15) 
     5  6 :                      Foam (20)                 Aluminium (18) 
     6  7 :                       Air (15)                       Air (15) 
     7  8 :                 DeadWater ( 8)                      Rock (16) 
     8  9 :                     Tyvek (25)                 DeadWater ( 8) 
     9 10 :                  OwsWater ( 9)                     Tyvek (25) 
    10 11 :                     Tyvek (25)                  OwsWater ( 9) 
    11 12 :                  IwsWater ( 6)                  IwsWater ( 6) 
    12 13 :            StainlessSteel (12)                  IwsWater ( 6) 
    13 14 :                MineralOil ( 4)            StainlessSteel (12) 
    14 15 :                   Acrylic ( 3)                MineralOil ( 4) 
    15 16 :        LiquidScintillator ( 2)                   Acrylic ( 3) 
    16 17 :                   Acrylic ( 3)        LiquidScintillator ( 2) 
    17 18 :                 GdDopedLS ( 1)                   Acrylic ( 3) 
    18 19 :                 GdDopedLS ( 1)        LiquidScintillator ( 2) 
    19 20 :                     Pyrex (14)                MineralOil ( 4) 
    20 21 :                    Vacuum (13)                     Pyrex (14) 
    21 22 :                  Bialkali ( 5)                    Vacuum (13) 
    22 23 :        UnstStainlessSteel (11)                MineralOil ( 4) 
    23 24 :                    Vacuum (13)                MineralOil ( 4) 



Suspect that shuffling was not done early enough to be reflected in the boundarylib/optical buffer::

    simon:ggeo blyth$ ./GBoundaryLibMetadata.py 
    INFO:__main__:['./GBoundaryLibMetadata.py']
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
      0 :  1 :                      Vacuum                    Vacuum                         -                         - 
      1 :  2 :                        Rock                    Vacuum                         -                         - 
      2 :  3 :                         Air                      Rock                         -                         - 
      3 :  4 :                         PPE                       Air                         - __dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface 
      4 :  5 :                   Aluminium                       Air                         -                         - 
      5 :  6 :                        Foam                 Aluminium                         -                         - 
      6 :  7 :                         Air                       Air                         -                         - 
      7 :  8 :                   DeadWater                      Rock                         -                         - 
      8 :  9 :                       Tyvek                 DeadWater                         - __dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface 
      9 : 10 :                    OwsWater                     Tyvek __dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface                         - 
     10 : 11 :                       Tyvek                  OwsWater                         -                         - 
     11 : 12 :                    IwsWater                  IwsWater                         -                         - 
     12 : 13 :              StainlessSteel                  IwsWater                         - __dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1 
     13 : 14 :                  MineralOil            StainlessSteel __dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface                         - 
     14 : 15 :                     Acrylic                MineralOil                         -                         - 
     15 : 16 :          LiquidScintillator                   Acrylic                         -                         - 
     16 : 17 :                     Acrylic        LiquidScintillator                         -                         - 
     17 : 18 :                   GdDopedLS                   Acrylic                         -                         - 
     18 : 19 :                   GdDopedLS        LiquidScintillator                         -                         - 
     19 : 20 :                       Pyrex                MineralOil                         -                         - 





Material index mapping 
------------------------

Cerenkov steps contain indices which have been chroma mapped already::

    In [4]: cs = CerenkovStep.get(1)

    In [8]: cs.materialIndices
    Out[8]: CerenkovStep([ 1,  8, 10, 12, 13, 14, 19], dtype=int32)

    In [9]: cs.materialIndex
    Out[9]: CerenkovStep([12, 12, 12, ...,  8,  8,  8], dtype=int32)

    In [11]: np.unique(cs.materialIndex)
    Out[11]: CerenkovStep([ 1,  8, 10, 12, 13, 14, 19], dtype=int32)


Modified daechromacontext.py to dump the chroma mapping with 
names translated into geant4 style::

    In [13]: import json

    In [15]: cmm = json.load(file("/tmp/ChromaMaterialMap.json"))

    In [16]: cmm
    Out[16]: 
    {u'/dd/Materials/ADTableStainlessSteel': 0,
     u'/dd/Materials/Acrylic': 1,
     u'/dd/Materials/Air': 2,
     u'/dd/Materials/Aluminium': 3,
     u'/dd/Materials/BPE': 4,
     u'/dd/Materials/Bialkali': 5,
     u'/dd/Materials/C_13': 6,
     u'/dd/Materials/Co_60': 7,
     u'/dd/Materials/DeadWater': 8,
     u'/dd/Materials/ESR': 9,
     u'/dd/Materials/GdDopedLS': 10,
     u'/dd/Materials/Ge_68': 11,
     u'/dd/Materials/IwsWater': 12,
     u'/dd/Materials/LiquidScintillator': 13,
     u'/dd/Materials/MineralOil': 14,
     u'/dd/Materials/Nitrogen': 15,
     u'/dd/Materials/NitrogenGas': 16,
     u'/dd/Materials/Nylon': 17,
     u'/dd/Materials/OpaqueVacuum': 18,
     u'/dd/Materials/OwsWater': 19,
     u'/dd/Materials/PVC': 20,
     u'/dd/Materials/Pyrex': 21,
     u'/dd/Materials/Silver': 22,
     u'/dd/Materials/StainlessSteel': 23,
     u'/dd/Materials/Teflon': 24,
     u'/dd/Materials/Tyvek': 25,
     u'/dd/Materials/UnstStainlessSteel': 26,
     u'/dd/Materials/Vacuum': 27,
     u'/dd/Materials/Water': 28}



::

    delta:npy blyth$ npy-g4stepnpy-test 
    Lookup::dump LookupTest 
    A   29 entries from /tmp/ChromaMaterialMap.json
    B   24 entries from /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GBoundaryLibMetadataMaterialMap.json
    A2B 21 entries in lookup  
      A    0 :     ADTableStainlessSteel  B  212 :     ADTableStainlessSteel 
      A    1 :                   Acrylic  B   56 :                   Acrylic 
      A    2 :                       Air  B    8 :                       Air 
      A    3 :                 Aluminium  B   16 :                 Aluminium 
      A    5 :                  Bialkali  B   84 :                  Bialkali 
      A    8 :                 DeadWater  B   28 :                 DeadWater 
      A    9 :                       ESR  B  104 :                       ESR 
      A   10 :                 GdDopedLS  B   68 :                 GdDopedLS 
      A   12 :                  IwsWater  B   44 :                  IwsWater 
      A   13 :        LiquidScintillator  B   60 :        LiquidScintillator 
      A   14 :                MineralOil  B   52 :                MineralOil 
      A   15 :                  Nitrogen  B  128 :                  Nitrogen 
      A   16 :               NitrogenGas  B  172 :               NitrogenGas 
      A   17 :                     Nylon  B  140 :                     Nylon 
      A   19 :                  OwsWater  B   36 :                  OwsWater 
      A   21 :                     Pyrex  B   76 :                     Pyrex 
      A   23 :            StainlessSteel  B   48 :            StainlessSteel 
      A   25 :                     Tyvek  B   32 :                     Tyvek 
      A   26 :        UnstStainlessSteel  B   88 :        UnstStainlessSteel 
      A   27 :                    Vacuum  B    0 :                    Vacuum 
      A   28 :                     Water  B  125 :                     Water 
    cs.dump
     ni 7836 nj 6 nk 4 nj*nk 24 
     (    0,    0)               -1                1               44               80  sid/parentId/materialIndex/numPhotons 
     (    0,    1)       -16536.295      -802084.812        -7066.000            0.844  position/time 
     (    0,    2)           -2.057            3.180            0.000            3.788  deltaPosition/stepLength 
     (    0,    3)               13           -1.000            1.000          299.791  code 
     (    0,    4)            1.000            0.000            0.000            0.719 
     (    0,    5)            0.482           79.201           79.201            0.000 
     ( 7835,    0)            -7836                1               28               48  sid/parentId/materialIndex/numPhotons 
     ( 7835,    1)       -20842.291      -795380.438        -7048.775           27.423  position/time 
     ( 7835,    2)           -1.068            1.669            0.004            1.981  deltaPosition/stepLength 
     ( 7835,    3)               13           -1.000            1.000          299.790  code 
     ( 7835,    4)            1.000            0.000            0.000            0.719 
     ( 7835,    5)            0.482           79.201           79.201            0.000 

    ... 28 [DeadWater] 
    ... 36 [OwsWater] 
    ... 44 [IwsWater] 
    ... 52 [MineralOil] 
    ... 56 [Acrylic] 
    ... 60 [LiquidScintillator] 
    ... 68 [GdDopedLS] 
    delta:npy blyth$ 

    (chroma_env)delta:env blyth$ ggeo-meta 28 36 44 52 56 60 68
    /usr/local/env/optix/ggeo/bin/GBoundaryLibTest /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae 28 36 44 52 56 60 68
    wavelength buffer NumBytes 134784 Ptr 0x10acde000 ItemSize 4 NumElements_PerItem 1 NumItems(NumBytes/ItemSize) 33696 NumElementsTotal (NumItems*NumElements) 33696 

    GBoundaryLib::dumpWavelengthBuffer wline 28 wsub 7 wprop 0 numSub 54 domainLength 39 numProp 16 

      28 |   7/  0 __dd__Materials__DeadWater0xbf8a548 
               1.390           1.390           1.372           1.357           1.352           1.346           1.341           1.335
             273.208         273.208        3164.640       12811.072       28732.207       13644.791        2404.398         371.974
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    GBoundaryLib::dumpWavelengthBuffer wline 36 wsub 9 wprop 0 numSub 54 domainLength 39 numProp 16 

      36 |   9/  0 __dd__Materials__OwsWater0xbf90c10 
               1.390           1.390           1.372           1.357           1.352           1.346           1.341           1.335
             273.208         273.208        3164.640       12811.072       28732.207       13644.791        2404.398         371.974
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    GBoundaryLib::dumpWavelengthBuffer wline 44 wsub 11 wprop 0 numSub 54 domainLength 39 numProp 16 

      44 |  11/  0 __dd__Materials__IwsWater0xc288f98 
               1.390           1.390           1.372           1.357           1.352           1.346           1.341           1.335
             273.208         273.208        3164.640       12811.072       28732.207       13644.791        2404.398         371.974
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    GBoundaryLib::dumpWavelengthBuffer wline 52 wsub 13 wprop 0 numSub 54 domainLength 39 numProp 16 

      52 |  13/  0 __dd__Materials__MineralOil0xbf5c830 
               1.434           1.758           1.540           1.488           1.471           1.464           1.459           1.457
              11.100          11.100          11.394        1078.898       24925.316       21277.369        5311.868         837.710
             850.000         850.000        4901.251       19819.381       52038.961      117807.406      252854.656      420184.219
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    GBoundaryLib::dumpWavelengthBuffer wline 56 wsub 14 wprop 0 numSub 54 domainLength 39 numProp 16 

      56 |  14/  0 __dd__Materials__Acrylic0xc02ab98 
               1.462           1.793           1.573           1.519           1.500           1.494           1.490           1.488
               0.008           0.008        4791.046        8000.000        8000.000        8000.000        8000.000        8000.000
             850.000         850.000        4901.251       19819.381       52038.961      117807.406      252854.656      420184.219
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    GBoundaryLib::dumpWavelengthBuffer wline 60 wsub 15 wprop 0 numSub 54 domainLength 39 numProp 16 

      60 |  15/  0 __dd__Materials__LiquidScintillator0xc2308d0 
               1.454           1.793           1.563           1.511           1.494           1.485           1.481           1.479
               0.001           0.001           0.198           1.913       26433.846       31710.930        6875.426         978.836
             850.000         850.000        4901.251       19819.381       52038.961      117807.406      252854.656      420184.219
               0.400           0.400           0.599           0.800           0.169           0.072           0.023           0.000
    GBoundaryLib::dumpWavelengthBuffer wline 68 wsub 17 wprop 0 numSub 54 domainLength 39 numProp 16 

      68 |  17/  0 __dd__Materials__GdDopedLS0xc2a8ed0 
               1.454           1.793           1.563           1.511           1.494           1.485           1.481           1.479
               0.001           0.001           0.198           1.913       26623.084       27079.125        7315.331         989.154
             850.000         850.000        4901.251       19819.381       52038.961      117807.406      252854.656      420184.219
               0.400           0.400           0.599           0.800           0.169           0.072           0.023           0.000


reemission_cdf reciprocation
-----------------------------

* for Geant4 match of photons generated in Chroma context needed to sample on 
  an energywise domain 1/wavelength[::-1]  

  * env/geant4/geometry/collada/collada_to_chroma.py::construct_cdf_energywise  

  * sampling wavelength-wise gives poor match at the extremes of the distribution 

  * i dont this there is anything fundamental here, its just matching precisely
    what is done by Geant4/NuWa generation

* this was implemented by special casing chroma material reemission_cdf property

env/geant4/geometry/collada/collada_to_chroma.py::

    515     def setup_cdf(self, material, props ):
    516         """
    517         Chroma uses "reemission_cdf" cumulative distribution function 
    518         to generate the wavelength of reemission photons. 
    519 
    520         Currently think that the name "reemission_cdf" is misleading, 
    521         as it is the RHS normalized CDF obtained from an intensity distribution
    522         (photon intensity as function of wavelength) 
    523 
    524         NB REEMISSIONPROB->reemission_prob is handled as a 
    525         normal keymapped property, no need to integrate to construct 
    526         the cdf for that.
    527     
    528         Compare this with the C++
    529 
    530            DsChromaG4Scintillation::BuildThePhysicsTable()  
    531 
    532         """ 

how to do this within ggeo/OptiX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. check what properties beyond the gang-of-four for materials and surfaces 
   are used by Chroma propagation

   * looks like makes sense to use a separate ggeo texture holding just reemission_cdf

/usr/local/env/chroma_env/src/chroma/chroma/cuda/geometry_types.h::

     04 struct Material
      5 {
      6     float *refractive_index;
      7     float *absorption_length;
      8     float *scattering_length;
      9     float *reemission_prob;
     10     float *reemission_cdf;   // SCB ? misleading as not only applicable to reemission ?  maybe intensity_cdf better
     11     unsigned int n;          // domain spec
     12     float step;              // domain spec
     13     float wavelength0;       // domain spec
     14 };
     ..
     18 struct Surface
     19 {
     20     float *detect;                 
     21     float *absorb;
     22     float *reemit;              // only used by propagate_at_wls 
     23     float *reflect_diffuse;
     24     float *reflect_specular;
     25     float *eta;                 // only used by propagate_complex
     26     float *k;                   // only used by propagate_complex
     27     float *reemission_cdf;      // only used by propagate_at_wls 
     28 
     29     unsigned int model;         // selects between SURFACE_DEFAULT, SURFACE_COMPLEX, SURFACE_WLS 
     30     unsigned int n;             // domain spec
     31     unsigned int transmissive;  // only used by propagate_complex
     32     float step;                 // domain spec
     33     float wavelength0;          // domain spec
     34     float thickness;A           // only used by propagate_complex
     35 };
        




Classes
--------

GGeo
    top level control and holder of other instances:
    GMesh GSolid GMaterial GSkinSurface GBorderSurface GBoundaryLib

GMesh
    holder of vertices, indices which fulfils GDrawable
    The GBuffer are created when setters like setVertices etc.. 
    are called 

    NB a relatively small number ~250 of GMesh instances are referenced
    from a much larger number ~12k of GNode arranged in the geometry tree 

    MUST THINK OF GMESH AS ABSTRACT SHAPES **NOT PLACED INSTANCES OF GEOMETRY**
    IT IS INCORRECT TO ASCRIBE SUBSTANCE OR NODE INDICES FOR EXAMPLE  
    SUCH THINGS BELONG ON THE GNODE


GMergedMesh
    specialization of GMesh that combines a tree of GNode 
    and referenced GNode shapes into a flattened single instance
    with transforms applied


GSolid
    GNode specialized with associated GBoundary and selection bit constituents

GNode
    identity index, GMesh and GMatrixF transform 
    also constituent unsigned int* arrays of length matching the face count

    m_substance_indices

    m_node_indices


GDrawable
    abstract interface definition returning GBuffer for vertices, normals, colors, texcoordinates, ...
    Only GMesh and GMergedMesh (by inheritance) fulfil the GDrawable interface

GBuffer
    holds pointer to some bytes together with integers describing the bytes : 
    
    nbytes 
           total number of bytes
    itemsize 
           size of each item in bytes
    nelem
           number of elements within each item 
           (eg item could be gfloat3 of itemsize 12, with nelem 3 
           corresponding to 3 floats) 

GBoundary
    holder of inner/outer material and inner/outer surface GPropertMaps

GBoundaryLib
    manager of substances, ensures duplicates are not created via digests


GBorderSurface
    PropertyMap specialization, specialization only used for creation
GSkinSurface
    PropertyMap specialization, specialization only used for creation
GMaterial
    PropertyMap specialization, specialization only used for creation
GPropertyMap
    ordered holder of GProperty<double> and GDomain<double>
GProperty<T>
    pair of GAry<T> for values and domain
GAry<T>
    array of values with linear interpolation functionality
GDomain
    standard range for properties, eg wavelength range and step


GVector
    gfloat3 guint3 structs
GMatrix
    4x4 matrix
GEnums
    material/surface property enums 

md5digest
    hashing


Where are substance indices formed and associated to every triangle ?
-----------------------------------------------------------------------

* indices are assigned by GBoundaryLib::get based on distinct property values
  this somewhat complicated approach is necessary as GBoundary incorporates info 
  from inner/outer material/surface so GBoundary 
  does not map to simple notions of identity it being a boundary between 
  materials with specific surfaces(or maybe no associated surface) 

* substance indices are affixed to the triangles of the geometry 
  by GSolid::setBoundary GNode::setBoundaryIndices
  which repeats the indice for every triangle of the solid. 
 
  This gets done within the AssimpGGeo::convertStructureVisit,
  the visitor method of the recursive AssimpGGeo::convertStructure 
  in assimpwrap-::

    506     GBoundaryLib* lib = gg->getBoundaryLib();
    507     GBoundary* boundary = lib->get(mt, mt_p, isurf, osurf, ...);
    510     solid->setBoundary(boundary);

* boundary indices are collected/flattened into 
  the unsigned int* substanceBuffer by GMergedMesh
       

How to map from Geant4 material indices into substance indices ?
--------------------------------------------------------------------

* chroma used a handshake to do this mapping using G4DAEChroma/G4DAEOpticks 
  communication : is this needed here ?

* ggeo is by design a dumb subtrate with which the geometry is represented, 
  the brains of ggeo creation are in assimpwrap-/AssimpGGeo especially:

  * AssimpGGeo::convertMaterials 
  * AssimpGGeo::addPropertyVector (note untested m_domain_reciprocal)



material code handshake between geant4<-->g4daechroma<-->chroma
------------------------------------------------------------------

Geant4/g4daechroma/chroma used metadata handshake resulting in 
a lookup table used by G4DAEChroma to convert geant4 material
codes into chroma ones, where is this implemented ?
 
* gdc-
* env/chroma/G4DAEChroma/G4DAEChroma/G4DAEMaterialMap.hh
* env/chroma/G4DAEChroma/src/G4DAEMaterialMap.cc 
* G4DAEChroma::SetMaterialLookup

* dsc- huh cant find this one locally 
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSimChroma/src 
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSimChroma/src/DsChromaRunAction_BeginOfRunAction.icc


G4/C++ side
~~~~~~~~~~~~~

DsChromaRunAction_BeginOfRunAction.icc::

    68      G4DAETransport* transport = new G4DAETransport(_transport.c_str());
    69      chroma->SetTransport( transport );
    70      chroma->Handshake();
    71  
    72      G4DAEMetadata* handshake = chroma->GetHandshake();
    73      //handshake->Print("DsChromaRunAction_BeginOfRunAction handshake");
    74  
    75      G4DAEMaterialMap* cmm = new G4DAEMaterialMap(handshake, "/chroma_material_map");
    76      chroma->SetMaterialMap(cmm);
    ..
    79  
    80  #ifndef NOT_NUWA
    81      // full nuwa environment : allows to obtain g4 material map from materials table
    82      G4DAEMaterialMap* gmm = new G4DAEMaterialMap();
    83  #else
    84      // non-nuwa : need to rely on handshake metadata for g4 material map
    85      G4DAEMaterialMap* gmm = new G4DAEMaterialMap(handshake, "/geant4_material_map");
    86  #endif
    87      //gmm->Print("#geant4_material_map");
    88  
    89      int* g2c = G4DAEMaterialMap::MakeLookupArray( gmm, cmm );
    90      chroma->SetMaterialLookup(g2c);
    91  

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSimChroma/src/DsChromaG4Cerenkov.cc

Lookup conversion applied as steps are collected (the most efficient place to do it)::

    308 #ifdef G4DAECHROMA_GPU_OPTICAL
    309     {
    310         // serialize DsG4Cerenkov::PostStepDoIt stack, just before the photon loop
    311         G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
    312         G4DAECerenkovStepList* csl = chroma->GetCerenkovStepList();
    313         int* g2c = chroma->GetMaterialLookup();
    314 
    315         const G4ParticleDefinition* definition = aParticle->GetDefinition(); 
    316         G4ThreeVector deltaPosition = aStep.GetDeltaPosition();
    317         G4double weight = fPhotonWeight*aTrack.GetWeight();
    318         G4int materialIndex = aMaterial->GetIndex();
    319 
    320         // this relates Geant4 materialIndex to the chroma equivalent
    321         G4int chromaMaterialIndex = g2c[materialIndex] ;

 
::

    104 void G4DAEChroma::Handshake(G4DAEMetadata* request)
    105 {
    106     if(!m_transport) return;
    107     m_transport->Handshake(request);
    108 }

    066 void G4DAETransport::Handshake(G4DAEMetadata* request)
     67 {
     68     if(!request) request = new G4DAEMetadata("{}");
     ..
     76     m_handshake = reinterpret_cast<G4DAEMetadata*>(m_socket->SendReceiveObject(request));
     ..
     85 }



python side
~~~~~~~~~~~~~~

Other end of that handshake:

* env/geant4/geometry/collada/g4daeview/daedirectpropagator.py 

::

     38     def incoming(self, request):
     39         """
     40         Branch handling based on itemshape (excluding first dimension) 
     41         of the request array 
     42         """
     43         self.chroma.incoming(request)  # do any config contained in request
     44         itemshape = request.shape[1:]
     ..
     54         elif itemshape == ():
     55 
     56             log.warn("empty itemshape received %s " % str(itemshape))
     57             extra = True
     ..
     76         return self.chroma.outgoing(response, results, extra=extra)

* env/geant4/geometry/collada/g4daeview/daechromacontext.py 

::

    224     def outgoing(self, response, results, extra=False):
    225         """
    226         :param response: NPL propagated photons
    227         :param results: dict of results from the propagation, eg times 
    228         """
    ...
    230         metadata = {}
    ...
    235         if extra:
    236             metadata['geometry'] = self.gpu_detector.metadata
    237             metadata['cpumem'] = self.mem.metadata()
    238             metadata['chroma_material_map'] = self.chroma_material_map
    239             metadata['geant4_material_map'] = self.geant4_material_map
    240         pass
    241         response.meta = [metadata]
    242         return response


Where does chroma_material_map get written ?
----------------------------------------------

* env/geant4/geometry/collada/g4daeview/daegeometry.py 

::

     796         cc = ColladaToChroma(DAENode, bvh=bvh )
     797         cc.convert_geometry(nodes=self.nodes())
     798 
     799         self.cc = cc
     800         self.chroma_material_map = DAEChromaMaterialMap( self.config, cc.cmm )
     801         self.chroma_material_map.write()
     802         log.debug("completed DAEChromaMaterialMap.write")


* env/geant4/geometry/collada/collada_to_chroma.py creates contiguous 0-based indices 
  for each unique material, the chroma array of materials then gets copied to GPU 
  hence the indices are as needed for GPU side material lookups

::

    634     def convert_make_maps(self):
    635         self.cmm = self.make_chroma_material_map( self.chroma_geometry )
    636         self.csm = self.make_chroma_surface_map( self.chroma_geometry )

::

    663     def make_chroma_material_map(self, chroma_geometry):
    664         """
    665         Curiously the order of chroma_geometry.unique_materials on different invokations is 
    666         "fairly constant" but not precisely so. 
    667         How is that possible ? Perfect or random would seem more likely outcomes. 
    668         """
    669         unique_materials = chroma_geometry.unique_materials
    670         material_lookup = dict(zip(unique_materials, range(len(unique_materials))))
    671         cmm = dict([(material_lookup[m],m.name) for m in filter(None,unique_materials)])
    672         cmm[-1] = "ANY"
    673         cmm[999] = "UNKNOWN"
    674         return cmm




GGeo Geometry Model Objective
------------------------------

* dumb holder of geometry information including the 
  extra material and surface properties, build on top of 
  mostly low level primitives with some use of map, string, vector,...  
  being admissable

  * NO imports from Assimp or OptiX
  * depend on user for most of the construction 

* intended to be a lightweight/slim intermediary format, eg  
  between raw Assimp geometry and properties
  to be consumed by the OptiX geometry creator/parameter setter.

* NB not trying to jump directly to an optix on GPU struct
  as the way to represent info efficiently within optix 
  needs experimentation : eg perhaps using texturemap lookups.
  Nevertheless, keep fairly low level to ease transition to
  on GPU structs

* intended to be a rather constant model, from which 
  a variety of OptiX representations can be developed 

* possible side-feature : geometry caching for fast starts
  without having to parse COLLADA.


Relationship to AssimpGeometry ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AssimpGGeo together with AssimpGeometry and AssimpTree 
(all from AssimpWrap) orchestrate creation of GGeo model
from the imported Assimp model. The OptiX model should then 
be created entirely from the GGeo model with no use of 
the Assimp model.

Analogs to Chroma
-------------------

* GMaterial : chroma.geometry.Material
* GSurface : chroma.geometry.Surface
* GMesh : chroma.geometry.Mesh
* GSolid : chroma.geometry.Solid
* GGeo : chroma.geometry.Geometry 

* AssimpWrap/AssimpGeometry/AssimpTree 
  are analogs of g4daenode.py and collada_to_chroma.py 


Basis Classes
---------------

::

    template <class T>
    class GProperty {

       * domain and value arrays + length

    class GPropertyMap {

       * string keyed map of GProperty<float>

    class GMaterial      : public GPropertyMap {
    class GBorderSurface : public GPropertyMap {
    class GSkinSurface   : public GPropertyMap {

    class GMesh {

        * vertices and faces


Client Classes
---------------

::

    class GSolid {

        * mesh + inside/outside materials and surfaces
        * nexus of structure

    class GGeo {

        * vectors of pointers to solids, materials, skin surfaces, border surfaces 


OptiX Geometry Model
---------------------

* many little programs and their parameters in flexible context 

* Material (closest hit program, anyhit program ) and params the programs use 
* Geometry (bbox program, intersection program) and params the programs use
* GeometryInstance associate Geometry with usually one Material (can be more than one) 

* try: representing material/surface props into 1D(wavelength) textures 

Chroma Geometry Model
----------------------

Single all encompassing Geometry instance containing:

* arrays of materials and surfaces
* material codes identifying material and surface indices for every triangle

chroma/chroma/cuda/geometry_types.h::

    struct Material
    {
        float *refractive_index;
        float *absorption_length;
        float *scattering_length;
        float *reemission_prob;
        float *reemission_cdf;   // SCB ? misleading as not only applicable to reemission ?  maybe intensity_cdf better
        unsigned int n;
        float step;
        float wavelength0;
    };

    struct Surface
    {
        float *detect;
        float *absorb;
        float *reemit;
        float *reflect_diffuse;
        float *reflect_specular;
        float *eta;
        float *k; 
        float *reemission_cdf;

        unsigned int model;
        unsigned int n;
        unsigned int transmissive;
        float step;
        float wavelength0;
        float thickness;
    };



    struct Geometry
    {
        float3 *vertices;
        uint3 *triangles;
        unsigned int *material_codes;
        unsigned int *colors;
        uint4 *primary_nodes;
        uint4 *extra_nodes;
        Material **materials;
        Surface **surfaces;
        float3 world_origin;
        float world_scale;
        int nprimary_nodes;
    };


Boundary Check for Cerenkov
------------------------------

::

    In [1]: a = oxc_(1)

    In [6]: count_unique(a[:,3,0].view(np.int32))
    Out[6]: 
    array([[    -1,  53472],
           [    11,  10062],   IwsWater/IwsWater
           [    12,   6612],   StainlessSteel/IwsWater
           [    13,   9021],   MineralOil/StainlessSteel
           [    14,  28583],   Acrylic/MineralOil
           [    15,  45059],   LiquidScintillator/Acrylic
           [    16,  95582],   Acrylic/LiquidScintillator
           [    17, 311100],   GdDopedLS/Acrylic
           [    19,    576],   Pyrex/MineralOil
           [    20,    282],   Vacuum/Pyrex
           [    22,   2776],   UnstStainlessSteel/MineralOil
           [    24,  36214],   Acrylic/MineralOil
           [    31,    710],   StainlessSteel/Water
           [    32,    194],   Nitrogen/StainlessSteel
           [    49,  11831],   UnstStainlessSteel/IwsWater
           [    50,     23],   Nitrogen/Water
           [    52,    744]])  Pyrex/IwsWater


::

    PhotonsNPY::classify
     17 :  17 :  311100 GdDopedLS.Acrylic..  
     16 :  16 :   95582 Acrylic.LiquidScintillator..  
     -1 :  -1 :   53472 unknown  
     15 :  15 :   45059 LiquidScintillator.Acrylic..  
     24 :  24 :   36214 Acrylic.MineralOil..RSOilSurface  
     14 :  14 :   28583 Acrylic.MineralOil..  
     49 :  49 :   11831 UnstStainlessSteel.IwsWater..AdCableTraySurface  
     11 :  11 :   10062 IwsWater.IwsWater..  
     13 :  13 :    9021 MineralOil.StainlessSteel.SSTOilSurface.  
     12 :  12 :    6612 StainlessSteel.IwsWater..SSTWaterSurfaceNear1  
     22 :  22 :    2776 UnstStainlessSteel.MineralOil..  
     52 :  52 :     744 Pyrex.IwsWater..  
     31 :  31 :     710 StainlessSteel.Water..  
     19 :  19 :     576 Pyrex.MineralOil..  
     20 :  20 :     282 Vacuum.Pyrex..  
     32 :  32 :     194 Nitrogen.StainlessSteel..  
     50 :  50 :      23 Nitrogen.Water..  



 
Sign of cos_theta
--------------------

For the major component GdDopedLS.Acrylic the sign coming out 

::

     34 RT_PROGRAM void closest_hit_propagate()
     35 {       
     ..
     38      const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     ..
     42      prd.cos_theta = dot(n,-ray.direction);





EOU
}

ggeo-env(){      elocal- ; opticks- ; }

ggeo-idir(){ echo $(opticks-idir); } 
ggeo-bdir(){ echo $(opticks-bdir)/$(ggeo-rel) ; }  

ggeo-sdir(){ echo $(env-home)/optix/ggeo ; }
ggeo-tdir(){ echo $(env-home)/optix/ggeo/tests ; }

ggeo-icd(){  cd $(ggeo-idir); }
ggeo-bcd(){  cd $(ggeo-bdir); }
ggeo-scd(){  cd $(ggeo-sdir); }

ggeo-cd(){  cd $(ggeo-sdir); }

ggeo-wipe(){
    local bdir=$(ggeo-bdir)
    rm -rf $bdir
}


ggeo-make(){
    local iwd=$PWD
    ggeo-bcd
    make $*
    cd $iwd
}

ggeo-install(){
   ggeo-make install
}


ggeo-bbin(){ echo $(ggeo-bdir)/GGeoTest ; }
ggeo-bin(){ echo $(ggeo-idir)/bin/${1:-GGeoTest} ; }


ggeo-export(){
    env | grep GGEO
}


ggeo-run(){
    ggeo-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

    ggeo-export 
    $DEBUG $(ggeo-bin) $*  
}

ggeo--(){
   # ggeo-cmake
   # ggeo-make
   # [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 
   # ggeo-install $*

   ( ggeo-bcd ; make ${1:-install} ) 
}

ggeo-lldb(){
    DEBUG=lldb ggeo-run
}

ggeo-brun(){
   echo running from bdir not idir : no install needed, but much set library path
   local bdir=$(ggeo-bdir)
   DYLD_LIBRARY_PATH=$bdir $DEBUG $bdir/GGeoTest 
}


ggeo-test(){
    local arg=$1
    ggeo-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

    ggeo-export $arg
    DEBUG=lldb ggeo-brun
}


ggeo-otool(){
   otool -L $(ggeo-bin)
}


   


ggeo-gpropertytest(){
   local bin=$(ggeo-bin GPropertyTest)
   echo $bin
   eval $bin
}


ggeo-ggeotest(){
   local bin=$(ggeo-bin GGeoTest)

   ggeoview-
   ggeoview-export

   echo $bin
   eval $bin $*
}

ggeo-mmt(){
   local bin=$(ggeo-bin GMergedMeshTest)

   ggeoview-
   ggeoview-export

   echo $bin
   eval $bin $*
}

ggeo-bbt(){
   local bin=$(ggeo-bin GBBoxMeshTest)

   ggeoview-
   ggeoview-export

   echo $bin
   eval $bin $*
}


ggeo-blt(){
   # formerly libtest 
   local bin=$(ggeo-bin GBoundaryLibTest)
   local cmd="$LLDB $bin $*"

   ggeoview-
   ggeoview-export

   echo $cmd
   eval $cmd
}


ggeo-blmt(){
   # formerly metatest 
   local bin=$(ggeo-bin GBoundaryLibMetadataTest)
   local cmd="$LLDB $bin $*"

   ggeoview-
   ggeoview-export

   echo $cmd
   eval $cmd
}


