opticks-full-CSGOptiX-build-error
===================================

::

     lo
     opticks-full



Looks like CSGOptiX build order issue::

    A[blyth@localhost issues]$ find /data1/blyth/local/opticks_Debug/build -name '*.so'
    /data1/blyth/local/opticks_Debug/build/okconf/libOKConf.so
    /data1/blyth/local/opticks_Debug/build/u4/libU4.so
    /data1/blyth/local/opticks_Debug/build/CSG/libCSG.so
    /data1/blyth/local/opticks_Debug/build/qudarap/libQUDARap.so
    /data1/blyth/local/opticks_Debug/build/sysrap/libSysRap.so
    /data1/blyth/local/opticks_Debug/build/g4cx/libG4CX.so
    /data1/blyth/local/opticks_Debug/build/gdxml/libGDXML.so
    A[blyth@localhost issues]$ 


::

    -- bcm_auto_pkgconfig_each LIB:CUDA::cudart : MISSING LIB_PKGCONFIG_NAME 
    -- Configuring done (2.3s)
    -- Generating done (0.0s)
    -- Build files have been written to: /data1/blyth/local/opticks_Debug/build/CSGOptiX
    === om-make-one : CSGOptiX        /home/blyth/opticks/CSGOptiX                                 /data1/blyth/local/opticks_Debug/build/CSGOptiX              
    [  1%] Building CUDA object CMakeFiles/CSGOptiX_OPTIX.dir/CSGOptiX7.ptx
    [  3%] Building CUDA object CMakeFiles/CSGOptiX_OPTIX.dir/Check.ptx
    [  5%] Generating opticks_CSGOptiX.pyi
    [  6%] Building CXX object CMakeFiles/nanobind-static.dir/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/src/nb_internals.cpp.o
    [  8%] Building CXX object CMakeFiles/nanobind-static.dir/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/src/nb_func.cpp.o
    [ 10%] Building CXX object CMakeFiles/nanobind-static.dir/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/src/nb_type.cpp.o
    [ 12%] Building CXX object CMakeFiles/nanobind-static.dir/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/src/nb_enum.cpp.o
    [ 13%] Building CXX object CMakeFiles/nanobind-static.dir/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/src/nb_ndarray.cpp.o
    [ 17%] Building CXX object CMakeFiles/nanobind-static.dir/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/src/nb_static_property.cpp.o
    [ 17%] Building CXX object CMakeFiles/nanobind-static.dir/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/src/nb_ft.cpp.o
    [ 20%] Building CXX object CMakeFiles/nanobind-static.dir/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/src/common.cpp.o
    [ 20%] Building CXX object CMakeFiles/nanobind-static.dir/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/src/error.cpp.o
    [ 24%] Building CXX object CMakeFiles/nanobind-static.dir/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/src/trampoline.cpp.o
    [ 24%] Building CXX object CMakeFiles/nanobind-static.dir/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/src/implicit.cpp.o
    Traceback (most recent call last):
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/stubgen.py", line 1488, in <module>
        main()
        ~~~~^^
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/nanobind/stubgen.py", line 1408, in main
        mod_imported = importlib.import_module(mod)
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/importlib/__init__.py", line 88, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
               ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
      File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
      File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
      File "<frozen importlib._bootstrap>", line 921, in _load_unlocked
      File "<frozen importlib._bootstrap>", line 813, in module_from_spec
      File "<frozen importlib._bootstrap_external>", line 1320, in create_module
      File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
    ImportError: libCSGOptiX.so: cannot open shared object file: No such file or directory
    make[2]: *** [CMakeFiles/opticks_CSGOptiX_stub.dir/build.make:73: opticks_CSGOptiX.pyi] Error 1
    make[1]: *** [CMakeFiles/Makefile2:994: CMakeFiles/opticks_CSGOptiX_stub.dir/all] Error 2
    make[1]: *** Waiting for unfinished jobs....
    [ 25%] Linking CXX static library libnanobind-static.a
    [ 25%] Built target nanobind-static
    [ 25%] Built target CSGOptiX_OPTIX
    make: *** [Makefile:146: all] Error 2
    === om-one-or-all cleaninstall : non-zero rc 2
    === om-all om-cleaninstall : ERROR bdir /data1/blyth/local/opticks_Debug/build/CSGOptiX : non-zero rc 2
    === om-one-or-all cleaninstall : non-zero rc 2
    === opticks-full : ERR from opticks-full-make
    (ok) A[blyth@localhost opticks]$ 


Fixed this by being more explicit which dependency control::

    204 #set(BUILD_WITH_NANOBIND OFF)
    205 set(BUILD_WITH_NANOBIND ON)
    206 if(BUILD_WITH_NANOBIND)
    207 
    208     find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
    209     execute_process(
    210       COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    211       OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    212 
    213     find_package(nanobind CONFIG)
    214     message(STATUS "${name} nanobind_ROOT  : ${nanobind_ROOT} ")
    215     message(STATUS "${name} nanobind_FOUND : ${nanobind_FOUND} ")
    216 
    217     set(py_ext opticks_${name})
    218 
    219     if(nanobind_FOUND)
    220        nanobind_add_module(${py_ext} NB_STATIC ${py_ext}.cc)
    221        target_link_libraries(${py_ext} PUBLIC ${name})
    222 
    223        # 1. Force order: lib -> python module -> stub
    224        add_dependencies(${py_ext} ${name})
    225 
    226        # 2. Add the stub generation
    227        nanobind_add_stub(
    228             ${py_ext}_stub
    229             MODULE ${py_ext}
    230             OUTPUT ${py_ext}.pyi
    231             PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR}
    232        )
    233 
    234 
    235        # 3. Explicitly link the stub target to the module to help order
    236        add_dependencies(${py_ext}_stub ${py_ext})
    237 
    238        # 4. CRITICAL: Tell the dynamic linker where to find libCSGOptiX.so during stubgen
    239        # This environment variable is used by the execute_process inside nanobind_add_stub
    240        set_property(TARGET ${py_ext}_stub PROPERTY
    241             ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{LD_LIBRARY_PATH}"
    242        )
    243 
    244 
    245        install(TARGETS ${py_ext} LIBRARY DESTINATION py)
    246        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${py_ext}.pyi DESTINATION py)
    247     endif()
    248 
    249 endif()


