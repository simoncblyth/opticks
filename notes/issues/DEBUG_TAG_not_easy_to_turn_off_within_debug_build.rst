DEBUG_TAG_not_easy_to_turn_off_within_debug_build
=====================================================

* need to get these tags 


CMake shared compile definitions for group of projects : Maybe CMake Presets ? Or just and included ".cmake" file ?
---------------------------------------------------------------------------------------------------------------------

* https://stackoverflow.com/questions/67385282/cmake-set-compile-options-and-compile-features-per-project/67388650
* CMakePresets.json
* https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html

* https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1?permalink_comment_id=3029072
* https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1

  Effective Modern CMake  


* https://stackoverflow.com/questions/47611319/target-compile-definitions-for-multiple-cmake-targets



DEBUG_TAG defined in multiple projects : TODO : central place for that
-------------------------------------------------------------------------------

HMM: maybe can just use fact that everything depends on sysrap/CMakeLists.txt

* so can just remove the settings from elsewhere ? 



::

    P[blyth@localhost qudarap]$ opticks-f DEBUG_TAG

    ./CSGOptiX/CMakeLists.txt:target_compile_definitions( ${name} PUBLIC DEBUG_TAG )



    ./qudarap/QSim__Desc.sh:DEBUG_TAG
    ./qudarap/CMakeLists.txt:DEBUG_TAG
    ./qudarap/CMakeLists.txt:      $<$<CONFIG:Debug>:DEBUG_TAG>

    ./sysrap/CMakeLists.txt:DEBUG_TAG 
    ./sysrap/CMakeLists.txt:#target_compile_definitions( ${name} PRIVATE DEBUG_TAG )    
    ./sysrap/CMakeLists.txt:      $<$<CONFIG:Debug>:DEBUG_TAG>

    ./u4/CMakeLists.txt:target_compile_definitions( ${name} PRIVATE DEBUG_TAG ) 
    ./u4/tests/CMakeLists.txt:target_compile_definitions( ${TGT} PUBLIC DEBUG_TAG ) 





    ./qudarap/QSim.cc:#ifdef DEBUG_TAG
    ./qudarap/QSim.cc:       << "DEBUG_TAG"
    ./qudarap/QSim.cc:       << "NOT-DEBUG_TAG"
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:    printf("//propagate_at_boundary.DEBUG_TAG ctx.idx %d base %p base.pidx %d \n", ctx.idx, base, base->pidx  ); 
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)

    ./sysrap/SEvt.cc:    ctx.end();   // copies {seq,sup} into evt->{seq,sup}[idx] (and tag, flat when DEBUG_TAG)
    ./sysrap/sctx.h:#ifdef DEBUG_TAG
    ./sysrap/ssys__Desc.sh:DEBUG_TAG
    ./sysrap/ssys.h:#ifdef DEBUG_TAG
    ./sysrap/ssys.h:       << "DEBUG_TAG"
    ./sysrap/ssys.h:       << "NOT:DEBUG_TAG"


    ./u4/CustomBoundary.h:#ifdef DEBUG_TAG
    ./u4/CustomBoundary.h:#ifdef DEBUG_TAG
    ./u4/CustomBoundary.h:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.hh:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.hh:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.hh:#ifdef DEBUG_TAG 
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpAbsorption.cc://#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/U4Physics.hh:#ifdef DEBUG_TAG
    ./u4/U4Physics.hh:#ifdef DEBUG_TAG
    ./u4/U4RandomDirection.hh:#ifdef DEBUG_TAG
    ./u4/U4RandomDirection.hh:#ifdef DEBUG_TAG
    ./u4/U4RandomDirection.hh:#ifdef DEBUG_TAG
    ./u4/U4RandomTools.hh:#ifdef DEBUG_TAG
    ./u4/U4RandomTools.hh:#ifdef DEBUG_TAG


    ./u4/U4Physics.cc:#ifdef DEBUG_TAG
    ./u4/U4Physics.cc:#if defined(DEBUG_TAG)
    ./u4/U4Physics.cc:    ss << "DEBUG_TAG" << std::endl ; 
    ./u4/U4Physics.cc:    ss << "NOT:DEBUG_TAG" << std::endl ; 
    ./u4/U4Physics.cc:#ifdef DEBUG_TAG
    ./u4/U4Physics.cc:#ifdef DEBUG_TAG
    ./u4/U4Random.hh:#ifdef DEBUG_TAG
    ./u4/U4Random.cc:#ifdef DEBUG_TAG
    ./u4/U4Random.cc:#ifdef DEBUG_TAG
    P[blyth@localhost opticks]$ 




Where is PRODUCTION defined ?
--------------------------------

Just sysrap/CMakeLists.txt::

    725 DEBUG_TAG
    726    needed for random aligned running in multiple pkgs: sysrap, qudarap, u4
    727    however making this PUBLIC makes rebuilding real heavy
    728    so must rely on making coordinated switches when doing random aligned running
    729 
    730 PLOG_LOCAL
    731    changes visibility of plog external symbols, allowing better
    732    integration with packages (like junosw) that do not hide
    733    symbols by default
    734 
    735 
    736 # TRY USING BUILD_TYPE dependent flags with generator expression
    737 #target_compile_definitions( ${name} PUBLIC OPTICKS_SYSRAP )
    738 #target_compile_definitions( ${name} PUBLIC WITH_CHILD ) 
    739 #target_compile_definitions( ${name} PUBLIC PRODUCTION )
    740 #target_compile_definitions( ${name} PUBLIC PLOG_LOCAL ) 
    741 #target_compile_definitions( ${name} PRIVATE DEBUG_TAG )    
    742 
    743 #]=]
    744 
    745 
    746 if(Custom4_FOUND)
    747    target_compile_definitions( ${name} PUBLIC WITH_CUSTOM4 )
    748    target_include_directories( ${name} PUBLIC ${Custom4_INCLUDE_DIR})
    749 endif()
    750 
    751 target_compile_definitions( ${name}
    752     PUBLIC
    753       $<$<CONFIG:Debug>:CONFIG_Debug>
    754       $<$<CONFIG:RelWithDebInfo>:CONFIG_RelWithDebInfo>
    755       $<$<CONFIG:Release>:CONFIG_Release>
    756       $<$<CONFIG:MinSizeRel>:CONFIG_MinSizeRel>
    757 
    758       OPTICKS_SYSRAP 
    759       WITH_CHILD
    760       PLOG_LOCAL
    761       $<$<CONFIG:Debug>:DEBUG_TAG>
    762       $<$<CONFIG:Debug>:DEBUG_PIDX>
    763       $<$<CONFIG:Debug>:DEBUG_PIDXYZ>
    764       $<$<CONFIG:Release>:PRODUCTION>
    765 )
    766 
    767 
        




::

    P[blyth@localhost opticks]$ opticks-f PRODUCTION
    ./CSG/csg_intersect_leaf.h:#if !defined(PRODUCTION) && defined(DEBUG_RECORD)
    ./CSG/csg_intersect_leaf.h:#if !defined(PRODUCTION) && defined(DEBUG_CYLINDER)
    ./CSG/csg_intersect_leaf.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDXYZ)
    ./CSG/csg_intersect_leaf.h:#if !defined(PRODUCTION) && defined(CSG_EXTRA)
    ./CSG/csg_intersect_leaf.h:#if !defined(PRODUCTION) && defined(CSG_EXTRA)
    ./CSG/csg_intersect_leaf.h:#if !defined(PRODUCTION) && defined(DEBUG)
    ./CSG/csg_intersect_leaf.h:#if !defined(PRODUCTION) && defined(DEBUG_RECORD)
    ./CSG/csg_intersect_leaf.h:#if !defined(PRODUCTION) && defined(DEBUG)
    ./CSG/csg_intersect_leaf.h:#if !defined(PRODUCTION) && defined(CSG_EXTRA)
    ./CSG/csg_intersect_leaf.h:#if !defined(PRODUCTION) && defined(DEBUG_RECORD)
    ./CSGOptiX/CSGOptiX7.cu:    * ifndef PRODUCTION sctx::trace sctx::point record the propagation point-by-point 
    ./CSGOptiX/CSGOptiX7.cu:#ifndef PRODUCTION
    ./CSGOptiX/CSGOptiX7.cu://#if !defined(PRODUCTION) && defined(WITH_RENDER)
    ./CSGOptiX/CSGOptiX7.cu:#ifndef PRODUCTION
    ./CSGOptiX/CSGOptiX7.cu:#ifndef PRODUCTION
    ./CSGOptiX/CSGOptiX7.cu:#ifndef PRODUCTION
    ./CSGOptiX/CSGOptiX7.cu:#ifndef PRODUCTION
    ./CSGOptiX/CSGOptiX7.cu://#if !defined(PRODUCTION) && defined(WITH_SIMTRACE)
    ./optickscore/Opticks.cc:    ss << ( isProduction() ? " PRODUCTION" : " DEVELOPMENT" ) ;
    ./preprocessor.sh:   -DPRODUCTION \
    ./qudarap/QEvent.hh:#ifndef PRODUCTION
    ./qudarap/QSim__Desc.sh:PRODUCTION
    ./qudarap/tests/QEventTest.cc:#ifndef PRODUCTION
    ./qudarap/tests/QEventTest.cc:#ifndef PRODUCTION
    ./qudarap/tests/QEventTest.cc:#ifndef PRODUCTION
    ./qudarap/QU.cc:#ifndef PRODUCTION
    ./qudarap/QU.cc:#ifndef PRODUCTION
    ./qudarap/QU.cc:#ifndef PRODUCTION
    ./qudarap/QU.cc:#ifndef PRODUCTION
    ./qudarap/QSim.cc:#ifdef PRODUCTION
    ./qudarap/QSim.cc:       << "PRODUCTION"
    ./qudarap/QSim.cc:       << "NOT-PRODUCTION"
    ./qudarap/qsim.h:#ifndef PRODUCTION
    ./qudarap/qsim.h:#ifndef PRODUCTION
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(MOCK_CUDA_DEBUG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(MOCK_CUDA_DEBUG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(MOCK_CUDA_DEBUG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(MOCK_CUDA_DEBUG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(MOCK_CUDA_DEBUG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_LOGF)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#ifndef PRODUCTION
    ./qudarap/qsim.h:#ifndef PRODUCTION
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#ifndef PRODUCTION
    ./qudarap/qsim.h:#ifndef PRODUCTION
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX) 
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/QEvent.cc:#ifndef PRODUCTION
    ./qudarap/QEvent.cc:#ifndef PRODUCTION 
    ./qudarap/QEvent.cc:#ifndef PRODUCTION 
    ./qudarap/QEvent.cc:#ifndef PRODUCTION 
    ./qudarap/QEvent.cc:#ifndef PRODUCTION 
    ./qudarap/QEvent.cc:#ifndef PRODUCTION 
    ./qudarap/QEvent.cc:#ifndef PRODUCTION 
    ./qudarap/QEvent.cc:#ifndef PRODUCTION
    ./qudarap/QEvent.cc:#ifndef PRODUCTION
    ./qudarap/QEvent.cc:#ifndef PRODUCTION
    ./qudarap/QEvent.cc:#ifndef PRODUCTION
    ./qudarap/qcerenkov.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)

    ./sysrap/CMakeLists.txt:#target_compile_definitions( ${name} PUBLIC PRODUCTION )
    ./sysrap/CMakeLists.txt:      $<$<CONFIG:Release>:PRODUCTION>
    
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.cc:#ifndef PRODUCTION
    ./sysrap/SEvt.hh:#ifndef PRODUCTION
    ./sysrap/SEvt.hh:#ifndef PRODUCTION 
    ./sysrap/SEvt.hh:#ifndef PRODUCTION
    ./sysrap/sctx.h:PRODUCTION macro. 
    ./sysrap/sctx.h:#ifndef PRODUCTION
    ./sysrap/sctx.h:#ifndef PRODUCTION
    ./sysrap/sctx.h:    // NB these are heavy : important to test with and without PRODUCTION 
    ./sysrap/sctx.h:#ifndef PRODUCTION
    ./sysrap/sctx.h:#ifndef PRODUCTION
    ./sysrap/sevent.h://#if !defined(PRODUCTION)
    ./sysrap/sevent.h:#ifndef PRODUCTION
    ./sysrap/sevent.h:#ifndef PRODUCTION
    ./sysrap/storch.h:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./sysrap/ssys__Desc.sh:PRODUCTION
    ./sysrap/tests/sreport.py:        eg: 'CONFIG_Release PRODUCTION WITH_CHILD WITH_CUSTOM4 PLOG_LOCAL '
    ./sysrap/ssys.h:#ifdef PRODUCTION
    ./sysrap/ssys.h:       << "PRODUCTION"
    ./sysrap/ssys.h:       << "NOT:PRODUCTION"
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifndef PRODUCTION
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifndef PRODUCTION
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifndef PRODUCTION
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifndef PRODUCTION
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifndef PRODUCTION
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifndef PRODUCTION
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifndef PRODUCTION
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifndef PRODUCTION
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifndef PRODUCTION
    ./u4/InstrumentedG4OpBoundaryProcess.hh:#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ./u4/Local_DsG4Scintillation.cc:#ifndef PRODUCTION
    ./u4/Local_DsG4Scintillation.cc:#ifndef PRODUCTION
    ./u4/Local_DsG4Scintillation.cc:#ifndef PRODUCTION
    ./u4/Local_DsG4Scintillation.cc:#ifndef PRODUCTION
    ./u4/Local_DsG4Scintillation.cc:#ifndef PRODUCTION
    ./u4/Local_DsG4Scintillation.cc:#ifndef PRODUCTION
    ./u4/Local_DsG4Scintillation.cc:#ifndef PRODUCTION
    ./u4/Local_DsG4Scintillation.cc:#ifndef PRODUCTION
    ./u4/ShimG4OpAbsorption.cc:#ifndef PRODUCTION
    ./u4/ShimG4OpRayleigh.cc:#ifndef PRODUCTION
    ./u4/ShimG4OpRayleigh.cc:#ifndef PRODUCTION
    ./u4/ShimG4OpRayleigh.cc:#ifndef PRODUCTION
    ./u4/ShimG4OpRayleigh.cc:#ifndef PRODUCTION
    ./u4/ShimG4OpRayleigh.cc:#ifndef PRODUCTION
    ./u4/ShimG4OpRayleigh.cc:#ifndef PRODUCTION
    ./u4/U4RandomDirection.hh:#ifndef PRODUCTION
    ./u4/U4RandomDirection.hh:#ifndef PRODUCTION
    ./u4/U4RandomTools.hh:#ifndef PRODUCTION
    ./u4/U4Recorder.cc:#ifdef PRODUCTION
    ./u4/U4Recorder.cc:    ss << "PRODUCTION" << std::endl ; 
    ./u4/U4Recorder.cc:    ss << "NOT:PRODUCTION" << std::endl ; 
    ./u4/U4Recorder.cc:#ifndef PRODUCTION
    ./u4/U4Recorder.cc:#ifndef PRODUCTION
    ./u4/U4Recorder.cc:#ifndef PRODUCTION
    ./u4/U4Recorder.cc:#ifndef PRODUCTION
    ./u4/U4Recorder.cc:#ifndef PRODUCTION
    ./u4/U4Random.hh:#ifndef PRODUCTION
    ./u4/U4Random.cc:#ifndef PRODUCTION
    ./u4/U4Random.cc:#ifndef PRODUCTION
    ./u4/U4Random.cc:#ifndef PRODUCTION
    P[blyth@localhost opticks]$ 





