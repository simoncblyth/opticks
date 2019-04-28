examples-go-fails
======================

* a few of them pop up a window, requiring an ESCAPE
  (TODO: somehow restrict number of renderloops when testing, or when an envvar OPTICKS_TESTING is set )



::

    [blyth@localhost examples]$ date
    Sun Apr 28 14:12:38 CST 2019

    blyth@localhost examples]$ ./gogo.sh 
    ./Geant4/G4Minimal/go.sh                 
    ./Geant4/G4Minimal/go.sh                 127           SKIP : incomplete example
    ./Geant4/CerenkovMinimal/go.sh           
    ./Geant4/CerenkovMinimal/go.sh           134           FIXED : CerenkovMinimal-X4PhysicalVolume-init-assert-no-mlib.rst
    ./Geant4/GDMLMangledLVNames/go.sh         
    ./Geant4/GDMLMangledLVNames/go.sh        127           FIXED 
    ./UseGGeo/go.sh                          
    ./UseGGeo/go.sh                          0 
    ./UseUseCUDA/go.sh                       
    ./UseUseCUDA/go.sh                       0 
    ./UseGLM/go.sh                           
    ./UseGLM/go.sh                           0 
    ./UseUseG4/go.sh                         
    ./UseUseG4/go.sh                         0 
    ./UseGLMRaw/go.sh                        
    ./UseGLMRaw/go.sh                        0 
    ./UseUseG4DAE/go.sh                      
    ./UseUseG4DAE/go.sh                      0 
    ./UseGeant4/go.sh                        
    ./UseGeant4/go.sh                        0 
    ./UseUseGLM/go.sh                        
    ./UseUseGLM/go.sh                        0 
    ./UseImGui/go.sh                         
    ./UseImGui/go.sh                         0 
    ./UseUseGLMRaw/go.sh                     
    ./UseUseGLMRaw/go.sh                     0 
    ./UseImplicitMesher/go.sh                
    ./UseImplicitMesher/go.sh                0 
    ./UseUseSysRap/go.sh                     
    ./UseUseSysRap/go.sh                     0 
    ./UseNPY/go.sh                           
    ./UseNPY/go.sh                           0 
    ./UseYoctoGL/go.sh                       
    ./UseYoctoGL/go.sh                       0 
    ./UseOGLRap/go.sh                        
    ./UseOGLRap/go.sh                        139        FIXED BY ADDING RENDERLOOPLIMIT
    ./UseOK/go.sh                            
    ./UseOK/go.sh                            0 
    ./UseYoctoGLRap/go.sh                    
    ./UseYoctoGLRap/go.sh                    127        SKIP : API has changed beneath this
    ./UseOKG4/go.sh                          
    ./UseOKG4/go.sh                          0 
    ./go.sh                                  
    ./go.sh                                  0 
    ./UseOKOP/go.sh                          
    ./UseOKOP/go.sh                          2          FIXED 
    ./UseOpenMesh/go.sh                      
    ./UseOpenMesh/go.sh                      2          FIXED
    ./UseOpenMeshRap/go.sh                   
    ./UseOpenMeshRap/go.sh                   0 
    ./UseOptiX/go.sh                         
    ./UseOptiX/go.sh                         0 
    ./UseOptiXRap/go.sh                      
    ./UseOptiXRap/go.sh                      139        FIXED
    ./UseOpticksAssimp/go.sh                 
    ./UseOpticksAssimp/go.sh                 0 
    ./UseOpticksBoost/go.sh                  
    ./UseOpticksBoost/go.sh                  2          ####################### CMake FindOpticksBoost issue
    ./UseAssimpRap/go.sh                     
    ./UseAssimpRap/go.sh                     0 
    ./UseOpticksCUDA/go.sh                   
    ./UseOpticksCUDA/go.sh                   0 
    ./UseOpticksCore/go.sh                   
    ./UseOpticksCore/go.sh                   139            
    ./UseBCM/go.sh                           
    ./UseBCM/go.sh                           0 
    ./UseBoost/go.sh                         
    ./UseBoost/go.sh                         0 
    ./UseOpticksGL/go.sh                     
    ./UseOpticksGL/go.sh                     139 
    ./UseOpticksGLEW/go.sh                   
    ./UseOpticksGLEW/go.sh                   0 
    ./UseBoostOld/go.sh                      
    ./UseBoostOld/go.sh                      2 
    ./UseOpticksGLFW/go.sh                   
    ./UseOpticksGLFW/go.sh                   0 
    ./UseBoostRap/go.sh                      
    ./UseBoostRap/go.sh                      2 
    ./UseOpticksGeo/go.sh                    
    ./UseOpticksGeo/go.sh                    0 
    ./UseCFG4/go.sh                          
    ./UseCFG4/go.sh                          2 
    ./UseOpticksXercesC/go.sh                
    ./UseOpticksXercesC/go.sh                0 
    ./UseCSGBSP/go.sh                        
    ./UseCSGBSP/go.sh                        2 
    ./UsePLog/go.sh                          
    ./UsePLog/go.sh                          0 
    ./UseCUDA/go.sh                          
    ./UseCUDA/go.sh                          0 
    ./UseSysRap/go.sh                        
    ./UseSysRap/go.sh                        0 
    ./UseCUDARap/go.sh                       
    ./UseCUDARap/go.sh                       0 
    ./UseThrustRap/go.sh                     
    ./UseThrustRap/go.sh                     0 
    ./UseCUDARapThrust/go.sh                 
    ./UseCUDARapThrust/go.sh                 0 
    ./UseUseBoost/go.sh                      
    ./UseUseBoost/go.sh                      127 
    ./UseDualContouringSample/go.sh          
    ./UseDualContouringSample/go.sh          0 
    ./UseG4/go.sh                            
    ./UseG4/go.sh                            0 
    ./UseG4DAE/go.sh                         
    ./UseG4DAE/go.sh                         2 
    ./UseG4OK/go.sh                          
    ./UseG4OK/go.sh                          127 
    ./UseOKConf/go.sh                        
    ./UseOKConf/go.sh                        0 
    ./UseSymbol/go.sh                        
    ./UseSymbol/go.sh                        0 
    ./UseUseSymbol/go.sh                     
    ./UseUseSymbol/go.sh                     0 
    ./ThrustOpenGLInterop/go.sh              
    ./ThrustOpenGLInterop/go.sh              2 
    ./UseOptiXProgram/go.sh                  
    ./UseOptiXProgram/go.sh                  0 
    ./UseOpticksGLFWSPPM/go.sh               
    ./UseOpticksGLFWSPPM/go.sh               0 
    ./UseOpticksGLFWSnap/go.sh               
    ./UseOpticksGLFWSnap/go.sh               0 
    ./UseOptiXProgramPP/go.sh                
    ./UseOptiXProgramPP/go.sh                0 
    ./UseOptiXBufferPP/go.sh                 
    ./UseOptiXBufferPP/go.sh                 1 
    ./UseOContextBufferPP/go.sh              
    ./UseOContextBufferPP/go.sh              1 
    ./UseOptiXBuffer/go.sh                   
    ./UseOptiXBuffer/go.sh                   0 
    ./UseInstance/go.sh                      
    ./UseInstance/go.sh                      0 
    ./UseOGLRapMinimal/go.sh                 
    ./UseOGLRapMinimal/go.sh                 0 
    ./UseOpenGL/go.sh                        
    ./UseOpenGL/go.sh                        0 
    ./UseGeometryShader/go.sh                
    ./UseGeometryShader/go.sh                0 
    ./UseShader/go.sh                        
    ./UseShader/go.sh                        0 
    ./UseXercesC/go.sh                       
    ./UseXercesC/go.sh                       0 
    ./UseOptiXGeometry/go.sh                 
    ./UseOptiXGeometry/go.sh                 0 
    ./UseOptiXGeometryTriangles/go.sh        
    ./UseOptiXGeometryTriangles/go.sh        0 

