CInterpolationTest_fail_to_find_material_with_0_based_index_36
================================================================


Looks like an issue with GBndLibIndex.npy in the old geocache (OPTICKS_RESOURCE_LAYOUT 1), thats
not present in the ab- validated slot 104


::

    epsilon:issues blyth$ CInterpolationTest 
    2018-08-12 17:30:37.990 INFO  [14467739] [main@52] CInterpolationTest
    ...
    2018-08-12 17:30:38.697 INFO  [14467739] [CMaterialBridge::initMap@61]  nmat 36 m_g4toix.size() 36 m_ixtoname.size() 36
    2018-08-12 17:30:38.697 INFO  [14467739] [CMaterialBridge::dump@97] CMaterialBridge::CMaterialBridge g4toix.size 36
                               /dd/Materials/GdDopedLS         0                     GdDopedLS                            Gd
                      /dd/Materials/LiquidScintillator         1            LiquidScintillator                            LS
                                 /dd/Materials/Acrylic         2                       Acrylic                            Ac
                              /dd/Materials/MineralOil         3                    MineralOil                            MO
                                /dd/Materials/Bialkali         4                      Bialkali                            Bk
                                /dd/Materials/IwsWater         5                      IwsWater                            Iw
                                   /dd/Materials/Water         6                         Water                            Wt
                               /dd/Materials/DeadWater         7                     DeadWater                            Dw
                                /dd/Materials/OwsWater         8                      OwsWater                            Ow
                                     /dd/Materials/ESR         9                           ESR                            ES
                            /dd/Materials/OpaqueVacuum        10                  OpaqueVacuum                            OV
                                    /dd/Materials/Rock        11                          Rock                            Rk
                                  /dd/Materials/Vacuum        12                        Vacuum                            Vm
                                   /dd/Materials/Pyrex        13                         Pyrex                            Py
                                     /dd/Materials/Air        14                           Air                            Ai
                                     /dd/Materials/PPE        15                           PPE                            PP
                               /dd/Materials/Aluminium        16                     Aluminium                            Al
                   /dd/Materials/ADTableStainlessSteel        17         ADTableStainlessSteel                            AS
                                    /dd/Materials/Foam        18                          Foam                            Fo
                                /dd/Materials/Nitrogen        19                      Nitrogen                            Ni
                             /dd/Materials/NitrogenGas        20                   NitrogenGas                            NG
                                   /dd/Materials/Nylon        21                         Nylon                            Ny
                                     /dd/Materials/PVC        22                           PVC                            PV
                                   /dd/Materials/Tyvek        23                         Tyvek                            Ty
                                /dd/Materials/Bakelite        24                      Bakelite                      Bakelite
                                  /dd/Materials/MixGas        25                        MixGas                        MixGas
                                    /dd/Materials/Iron        26                          Iron                          Iron
                                  /dd/Materials/Teflon        27                        Teflon                        Teflon
                      /dd/Materials/UnstStainlessSteel        28            UnstStainlessSteel                            US
                                     /dd/Materials/BPE        29                           BPE                           BPE
                                   /dd/Materials/Ge_68        30                         Ge_68                         Ge_68
                                   /dd/Materials/Co_60        31                         Co_60                         Co_60
                                    /dd/Materials/C_13        32                          C_13                          C_13
                                  /dd/Materials/Silver        33                        Silver                        Silver
                                 /dd/Materials/RadRock        34                       RadRock                       RadRock
                          /dd/Materials/StainlessSteel        35                StainlessSteel                            SS


  ...

    2018-08-12 17:30:38.796 INFO  [14467739] [main@190]    17( 5,-1,-1, 5)                                         IwsWater///IwsWater om         /dd/Materials/IwsWater im         /dd/Materials/IwsWater
    2018-08-12 17:30:38.796 INFO  [14467739] [main@141]  i  18 omat   5 osur   4 isur 4294967295 imat  36
    2018-08-12 17:30:38.796 FATAL [14467739] [*CMaterialBridge::getG4Material@190]  failed to find a G4Material with index 36 in all the indices 15 25 14 24 18 16 26 22 0 2 27 1 4 10 12 13 28 35 9 21 3 29 30 31 32 33 19 6 20 5 17 23 8 7 34 11 
    Assertion failed: (im), function main, file /Users/blyth/opticks/cfg4/tests/CInterpolationTest.cc, line 152.
    Abort trap: 6
    epsilon:issues blyth$ 

::

    In [1]: a = np.load("GBndLibIndex.npy")

    In [5]: a[:,0].min()
    Out[5]: 1

    In [6]: a[:,0].max()
    Out[6]: 36

    In [7]: a[:,3].max()
    Out[7]: 36

    In [8]: a[:,3].min()
    Out[8]: 0

    In [9]: pwd
    Out[9]: u'/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GBndLib'


vs 104::

    In [4]: a[:,0].min()
    Out[4]: 1

    In [5]: a[:,0].max()
    Out[5]: 35

    In [6]: a[:,3].max()
    Out[6]: 35

    In [7]: a[:,3].min()
    Out[7]: 0

    In [8]: pwd
    Out[8]: u'/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/104/GBndLib'







    epsilon:issues blyth$ OPTICKS_RESOURCE_LAYOUT=104 CInterpolationTest 
    2018-08-12 17:33:45.482 INFO  [14469690] [main@52] CInterpolationTest
    ...
    2018-08-12 17:33:46.183 INFO  [14469690] [CMaterialBridge::initMap@61]  nmat 36 m_g4toix.size() 36 m_ixtoname.size() 36
    2018-08-12 17:33:46.183 INFO  [14469690] [CMaterialBridge::dump@97] CMaterialBridge::CMaterialBridge g4toix.size 36
                               /dd/Materials/GdDopedLS         0                     GdDopedLS                            Gd
                      /dd/Materials/LiquidScintillator         1            LiquidScintillator                            LS
                                 /dd/Materials/Acrylic         2                       Acrylic                            Ac
                              /dd/Materials/MineralOil         3                    MineralOil                            MO
                                /dd/Materials/Bialkali         4                      Bialkali                            Bk
                                /dd/Materials/IwsWater         5                      IwsWater                            Iw
                                   /dd/Materials/Water         6                         Water                            Wt
                               /dd/Materials/DeadWater         7                     DeadWater                            Dw
                                /dd/Materials/OwsWater         8                      OwsWater                            Ow
                                     /dd/Materials/ESR         9                           ESR                            ES
                            /dd/Materials/OpaqueVacuum        10                  OpaqueVacuum                            OV
                                    /dd/Materials/Rock        11                          Rock                            Rk
                                  /dd/Materials/Vacuum        12                        Vacuum                            Vm
                                   /dd/Materials/Pyrex        13                         Pyrex                            Py
                                     /dd/Materials/Air        14                           Air                            Ai
                                     /dd/Materials/PPE        15                           PPE                            PP
                               /dd/Materials/Aluminium        16                     Aluminium                            Al
                   /dd/Materials/ADTableStainlessSteel        17         ADTableStainlessSteel                            AS
                                    /dd/Materials/Foam        18                          Foam                            Fo
                                /dd/Materials/Nitrogen        19                      Nitrogen                            Ni
                             /dd/Materials/NitrogenGas        20                   NitrogenGas                            NG
                                   /dd/Materials/Nylon        21                         Nylon                            Ny
                                     /dd/Materials/PVC        22                           PVC                            PV
                                   /dd/Materials/Tyvek        23                         Tyvek                            Ty
                                /dd/Materials/Bakelite        24                      Bakelite                      Bakelite
                                  /dd/Materials/MixGas        25                        MixGas                        MixGas
                                    /dd/Materials/Iron        26                          Iron                          Iron
                                  /dd/Materials/Teflon        27                        Teflon                        Teflon
                      /dd/Materials/UnstStainlessSteel        28            UnstStainlessSteel                            US
                                     /dd/Materials/BPE        29                           BPE                           BPE
                                   /dd/Materials/Ge_68        30                         Ge_68                         Ge_68
                                   /dd/Materials/Co_60        31                         Co_60                         Co_60
                                    /dd/Materials/C_13        32                          C_13                          C_13
                                  /dd/Materials/Silver        33                        Silver                        Silver
                                 /dd/Materials/RadRock        34                       RadRock                       RadRock
                          /dd/Materials/StainlessSteel        35                StainlessSteel                            SS



