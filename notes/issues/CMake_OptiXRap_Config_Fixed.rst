CMake_OptiXRap_Config_Fixed
=============================



Issue reported by Axel : CMake fail for OptiXRap
---------------------------------------------------


I succeeded to reproduce the issue by cleaning and building optixrap with:

::

    epsilon:optixrap blyth$ om-clean | sh  
    epsilon:optixrap blyth$ om-conf
    === om-one-or-all conf : bdir /usr/local/opticks/build/optixrap does not exist : creating it
    === om-one-or-all conf : optixrap        /Users/blyth/opticks/optixrap                                /usr/local/opticks/build/optixrap                            
    ...
    -- Configuring OptiXRap
    -- /Users/blyth/opticks/cmake/Modules/FindOptiX.cmake : OptiX_VERBOSE     : ON 
    -- /Users/blyth/opticks/cmake/Modules/FindOptiX.cmake : OptiX_INSTALL_DIR :  
    CMake Error at /Users/blyth/opticks/cmake/Modules/FindOptiX.cmake:84 (message):
      optix library not found.  Try adding cmake argument:
      -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) (all packages downstream
      from OptiX need this)
    Call Stack (most recent call first):
      /Users/blyth/opticks/cmake/Modules/FindOptiX.cmake:96 (OptiX_report_error)
      CMakeLists.txt:7 (find_package)


    -- Configuring incomplete, errors occurred!
    See also "/usr/local/opticks/build/optixrap/CMakeFiles/CMakeOutput.log".
    === om-one-or-all conf : non-zero rc 1
    epsilon:optixrap blyth$ 


Presumably my CMake cache for optixrap had the values it it, from prior to some CMake 
reorganisation so it was not effecting me before, until I did the clean. 
Which has allowed me to fix it, it was simply a missing  find_package for OKConf in optixrap, 

::

     find_package(OKConf     REQUIRED CONFIG)

Actually its possible to see that from the dependencies tree:

::

    epsilon:Modules blyth$ opticks-deps
    INFO:__main__:root /Users/blyth/opticks 
     10          OKCONF :               okconf :               OKConf : OpticksCUDA OptiX G4  
     20          SYSRAP :               sysrap :               SysRap : OKConf PLog  
     30            BRAP :             boostrap :             BoostRap : Boost PLog SysRap  
     40             NPY :                  npy :                  NPY : PLog GLM OpenMesh BoostRap YoctoGL ImplicitMesher DualContouringSample  
     45             YOG :           yoctoglrap :           YoctoGLRap : NPY  
     50          OKCORE :          optickscore :          OpticksCore : NPY  
     60            GGEO :                 ggeo :                 GGeo : OpticksCore YoctoGLRap  
     70          ASIRAP :            assimprap :            AssimpRap : OpticksAssimp GGeo  
     80         MESHRAP :          openmeshrap :          OpenMeshRap : GGeo OpticksCore  
     90           OKGEO :           opticksgeo :           OpticksGeo : OpticksCore AssimpRap OpenMeshRap  
    100         CUDARAP :              cudarap :              CUDARap : SysRap OpticksCUDA  
    110           THRAP :            thrustrap :            ThrustRap : OpticksCore CUDARap  

    120           OXRAP :             optixrap :             OptiXRap : OKConf OptiX OpticksGeo ThrustRap    ###### OKConf added ####### 
                                                                       ^^^^^^^^^

    130            OKOP :                 okop :                 OKOP : OptiXRap  
    140          OGLRAP :               oglrap :               OGLRap : ImGui OpticksGLEW OpticksGLFW OpticksGeo  
    150            OKGL :            opticksgl :            OpticksGL : OGLRap OKOP  
    160              OK :                   ok :                   OK : OpticksGL  
    165              X4 :                extg4 :                ExtG4 : G4 GGeo  
    170            CFG4 :                 cfg4 :                 CFG4 : G4 ExtG4 OpticksXercesC OpticksGeo  
    180            OKG4 :                 okg4 :                 OKG4 : OK CFG4  
    190            G4OK :                 g4ok :                 G4OK : CFG4 ExtG4 OKOP  
    epsilon:Modules blyth$ 


In optixrap,  OptiX was preceded by nothing, which won’t work it must be preceded by OKConf directly 
or some other package that depends on OKConf. 


