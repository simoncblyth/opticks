opticks-examples-gogo-fails
==============================

::

    blyth@localhost examples]$ ./gogo.sh 
    ./Geant4/G4Minimal/go.sh                 SKIP 
    ./Geant4/CerenkovMinimal/go.sh           0 
    ./Geant4/GDMLMangledLVNames/go.sh        0 
    ./UseGGeo/go.sh                          0 
    ./UseUseCUDA/go.sh                       0 
    ./UseGLM/go.sh                           0 
    ./UseUseG4/go.sh                         0 
    ./UseGLMRaw/go.sh                        0 
    ./UseUseG4DAE/go.sh                      0 
    ./UseGeant4/go.sh                        0 
    ./UseUseGLM/go.sh                        0 
    ./UseImGui/go.sh                         0 
    ./UseUseGLMRaw/go.sh                     0 
    ./UseImplicitMesher/go.sh                0 
    ./UseUseSysRap/go.sh                     0 
    ./UseNPY/go.sh                           0 
    ./UseYoctoGL/go.sh                       0 
    ./UseOGLRap/go.sh                        0 
    ./UseOK/go.sh                            0 
    ./UseYoctoGLRap/go.sh                    SKIP 
    ./UseOKG4/go.sh                          0 
    ./go.sh                                  0 
    ./UseOKOP/go.sh                          0 
    ./UseOpenMesh/go.sh                      0 
    ./UseOpenMeshRap/go.sh                   0 
    ./UseOptiX/go.sh                         0 
    ./UseOptiXRap/go.sh                      0 
    ./UseOpticksAssimp/go.sh                 0 
    ./UseOpticksBoost/go.sh                  127     DELETE : have moved to direct Boost in boostrap, so delete this
    ./UseAssimpRap/go.sh                     0 
    ./UseOpticksCUDA/go.sh                   0 
    ./UseOpticksCore/go.sh                   139     FIXED: resource setup moved 
    ./UseBCM/go.sh                           0 
    ./UseBoost/go.sh                         0 
    ./UseOpticksGL/go.sh                     139     FIXED : Add OPTICKS_DEFAULT_INTEROP_CVD to avoid cvd fail, also needed argforced    --renderlooplimit 2000    
    ./UseOpticksGLEW/go.sh                   0 
    ./UseBoostOld/go.sh                      2       FIXED : missed -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals  causing cmake to fail to find BCM 
    ./UseOpticksGLFW/go.sh                   0 
    ./UseBoostRap/go.sh                      2       FIXED : ditto
    ./UseOpticksGeo/go.sh                    0 
    ./UseCFG4/go.sh                          2       FIXED : old CGeometry interface, missed CSensitiveDetector 
    ./UseOpticksXercesC/go.sh                0 
    ./UseCSGBSP/go.sh                        2       SKIP : CSGBSP not found, just meshing expt   
    ./UsePLog/go.sh                          0 
    ./UseCUDA/go.sh                          0 
    ./UseSysRap/go.sh                        0 
    ./UseCUDARap/go.sh                       0 
    ./UseThrustRap/go.sh                     0 
    ./UseCUDARapThrust/go.sh                 0 
    ./UseUseBoost/go.sh                      127     FIXED : boost finding cmake command line : less is more 
    ./UseDualContouringSample/go.sh          0 
    ./UseG4/go.sh                            0 
    ./UseG4DAE/go.sh                         2       SKIP : this is legacy now 
    ./UseG4OK/go.sh                          127     FIXED : old classname, CMake ppolicy warnings 
    ./UseOKConf/go.sh                        0 
    ./UseSymbol/go.sh                        0 
    ./UseUseSymbol/go.sh                     0 
    ./ThrustOpenGLInterop/go.sh              2       nvcc not finding gl symbols
    ./UseOptiXProgram/go.sh                  0 
    ./UseOpticksGLFWSPPM/go.sh               0 
    ./UseOpticksGLFWSnap/go.sh               0 
    ./UseOptiXProgramPP/go.sh                0 
    ./UseOptiXBufferPP/go.sh                 0 
    ./UseOptiXGeometryStandalone/go.sh       0 
    ./UseOContextBufferPP/go.sh              0 
    ./UseOptiXBuffer/go.sh                   127     FIXED : missed array without NPY 
    ./UseInstance/go.sh                      0 
    ./UseOGLRapMinimal/go.sh                 134     FIXED : needed Opticks instance
    ./UseOpenGL/go.sh                        0 
    ./UseGeometryShader/go.sh                0 
    ./UseShader/go.sh                        0 
    ./UseXercesC/go.sh                       0 
    ./UseOptiXGeometry/go.sh                 0 
    ./UseOptiXGeometryTriangles/go.sh        2       
    ./UseOptiXGeometryInstancedStandalone/go.sh 0 

