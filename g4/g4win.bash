# === func-gen- : g4/g4win fgp g4/g4win.bash fgn g4win fgh g4
g4win-src(){      echo g4/g4win.bash ; }
g4win-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4win-src)} ; }
g4win-vi(){       vi $(g4win-source) ; }
g4win-env(){      elocal- ; }
g4win-usage(){ cat << EOU

G4 On Windows notes
=====================






MSVC Passing strings incompatibility
--------------------------------------

* http://programmers.stackexchange.com/questions/176681/did-c11-address-concerns-passing-std-lib-objects-between-dynamic-shared-librar


MSVC iterator debugging
-------------------------

* https://msdn.microsoft.com/en-us/library/aa985982.aspx

See: cfg4/tests/G4BoxTest.cc

The nasty dialog box entitled "Microsoft Visual C++ Runtime Library" 
can be avoided with the define::

    #define _HAS_ITERATOR_DEBUGGING 0

This also allows to pass strings to G4. 


_ITERATOR_DEBUG_LEVEL
------------------------


Because the _SECURE_SCL and _HAS_ITERATOR_DEBUGGING macros support similar
functionality, users are often uncertain which macro and macro value to use in
a particular situation. To resolve this issue, we recommend that you use only
the _ITERATOR_DEBUG_LEVEL macro.

* https://msdn.microsoft.com/en-us/library/hh697468.aspx




G4 Windows dllexport/dllimport ?
-----------------------------------

::

    delta:geant4.10.02 blyth$ find source -name '*.hh' -exec grep -H dll {} \;
    source/g3tog4/include/G3toG4Defs.hh:      #define G3G4DLL_API __declspec( dllexport )
    source/g3tog4/include/G3toG4Defs.hh:      #define G3G4DLL_API __declspec( dllimport )
    source/global/management/include/G4Types.hh:    #define G4DLLEXPORT __declspec( dllexport )
    source/global/management/include/G4Types.hh:    #define G4DLLIMPORT __declspec( dllimport )
    delta:geant4.10.02 blyth$ 

Huh almost no symbols exported::

    delta:geant4.10.02 blyth$ find source -name '*.hh' -exec grep -H DLL {} \; | wc -l
         125

* does that mean must use static G4 on windows ?


config/genwindef.cc reads .dll writes .def
-------------------------------------------------

::

    ntuhep@ntuhep-PC MINGW64 ~/local/opticks/externals/g4/geant4_10_02_p01
    $ find . -name '*.txt'  -exec grep -H genwindef {} \;
    ./ReleaseNotes/Patch4.10.0-2.txt:        DLLs build on Windows with genwindef used also in CMake.

::

    $ find . -name '*.cmake'  -exec grep -H genwindef {} \;
    ./cmake/Modules/Geant4LibraryBuildOptions.cmake:# On WIN32, we need to build the genwindef application to create export
    ./cmake/Modules/Geant4LibraryBuildOptions.cmake:# if it can be protected so that the genwindef target wouldn't be defined
    ./cmake/Modules/Geant4LibraryBuildOptions.cmake:  get_filename_component(_genwindef_src_dir ${CMAKE_CURRENT_LIST_FILE} PATH)
    ./cmake/Modules/Geant4LibraryBuildOptions.cmake:  add_executable(genwindef EXCLUDE_FROM_ALL
    ./cmake/Modules/Geant4LibraryBuildOptions.cmake:    ${_genwindef_src_dir}/genwindef/genwindef.cpp
    ./cmake/Modules/Geant4LibraryBuildOptions.cmake:    ${_genwindef_src_dir}/genwindef/LibSymbolInfo.h
    ./cmake/Modules/Geant4LibraryBuildOptions.cmake:    ${_genwindef_src_dir}/genwindef/LibSymbolInfo.cpp)
    ./cmake/Modules/Geant4MacroLibraryTargets.cmake:        COMMAND genwindef -o _${G4LIBTARGET_NAME}-${CMAKE_CFG_INTDIR}.def -l ${G4LIBTARGET_NAME} $<TARGET_FILE:${_archive}>
    ./cmake/Modules/Geant4MacroLibraryTargets.cmake:        DEPENDS ${_archive} genwindef)
    ./cmake/Templates/UseGeant4_internal.cmake:    # - Use genwindef to create .def file listing symbols
    ./cmake/Templates/UseGeant4_internal.cmake:      COMMAND ${genwindef_cmd} -o ${library}.def -l ${library} ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}/${library}-arc.lib
    ./cmake/Templates/UseGeant4_internal.cmake:      DEPENDS ${library}-arc genwindef)



cmake/Templates/UseGeant4_internal.cmake::

     17 function(geant4_link_library library)
     18   cmake_parse_arguments(ARG "TYPE;LIBRARIES" "" ${ARGN})
     19   set(sources)
     20
     21   # - Fill sources
     22   foreach(fp ${ARG_UNPARSED_ARGUMENTS})
     23     if(IS_ABSOLUTE ${fp})
     24       file(GLOB files ${fp})
     25     else()
     26       file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${fp})
     27     endif()
     28     if(files)
     29       set(sources ${sources} ${files})
     30     else()
     31       set(sources ${sources} ${fp})
     32     endif()
     33   endforeach()
     34
     35   # - Shared library unless specified
     36   if(NOT ARG_TYPE)
     37     set(ARG_TYPE SHARED)
     38   endif()
     39
     40   # - Make sure we can access our own headers
     41   include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
     42
     43   # - Deal with Win32 DLLs that don't export via declspec
     44   if(WIN32 AND ARG_TYPE STREQUAL SHARED)
     45     # - Dummy archive library
     46     add_library( ${library}-arc STATIC EXCLUDE_FROM_ALL ${sources})

     47
     48     # - Use genwindef to create .def file listing symbols
     49     add_custom_command(
     50       OUTPUT ${library}.def
     51       COMMAND ${genwindef_cmd} -o ${library}.def -l ${library} ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}/${library}-arc.lib
     52       DEPENDS ${library}-arc genwindef)
     53
     54     #- Dummy cpp file needed to satisfy Visual Studio.
     55     file( WRITE ${CMAKE_CURRENT_BINARY_DIR}/${library}.cpp "// empty file\n" )
     56     add_library( ${library} SHARED ${library}.cpp ${library}.def)

     //     compile sources into static lib,
     //     pull out the symbols with genwindef to make .def  
     //     


     57     target_link_libraries(${library} ${library}-arc ${ARG_LIBRARIES})
     58     set_target_properties(${library} PROPERTIES LINK_INTERFACE_LIBRARIES ${ARG_LIBRARIES} ${Geant4_LIBRARIES})
     59   else()
     60     add_library( ${library} ${ARG_TYPE} ${sources})
     61     target_link_libraries(${library} ${ARG_LIBRARIES} ${Geant4_LIBRARIES})
     62   endif()
     63 endfunction()


So cmake consumes the .def and uses it to rustle up the .dll 


* https://blog.kitware.com/create-dlls-on-windows-without-declspec-using-new-cmake-export-all-feature/




Exporting from DLL using DEF files
-------------------------------------

* https://msdn.microsoft.com/en-us/library/d91k01sh.aspx

* http://stackoverflow.com/questions/6720655/linking-to-a-dll-with-a-def-file-instead-of-a-lib-file

* http://stackoverflow.com/questions/225432/export-all-symbols-when-creating-a-dll



building against G4, mentions windows but not DLLs
----------------------------------------------------

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/ch03s02.html


bin/G4global.dll (windows only)
-------------------------------------

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/ch03.html



G4 Forum for install/config issues
------------------------------------

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/installconfig.html

DLL Search Order
~~~~~~~~~~~~~~~~~~

* https://msdn.microsoft.com/en-us/library/windows/desktop/ms682586(v=vs.85).aspx


Portable G4
~~~~~~~~~~~~~~~

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/installconfig/1799/1.html

Ben Morgan 

At least for the Geant4 libraries, these can be linked statically to the
application and this will typically result in a smaller bundle (depending on
what physics processes the application uses). To link an application to static
Geant4 libraries rather than dynamic, ensure the install of Geant4 is done with
the cmake argument BUILD_STATIC_LIBS enabled. When building the application
with cmake, add the static component to the list of features needed in the call
to find_package:

::

    find_package(Geant4 REQUIRED static)

    add_executable(foo foo.cc)
    target_link_libraries(foo ${Geant4_LIBRARIES})

Here, the addition of the static argument ensures that the Geant4_LIBRARIES
variable is populated with static rather than dynamic arguments.


Comparison of static/dynamic linkage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.codeproject.com/Articles/85391/Microsoft-Visual-C-Static-and-Dynamic-Libraries

Interesting points:

* with static libs when you dont have a header you can use::

    extern int add(int a, int b);

* looks like using static libs avoids the need for dllexport/dllimport



MSVC Approach
----------------

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/ch01.html

Officially supported: Windows 7 with Visual Studio 2013 or 2015, 64bit.


* web/installer download of Visual Studio 2015 community 


Painful GUI Walkthru of installing G4 on Windows
--------------------------------------------------

* http://www2.warwick.ac.uk/fac/sci/physics/staff/research/bmorgan/geant4/installingonwindows/

Includes advice to individually set envvars via GUI panel.

MinGW/NSYS2 Approach 
------------------------

::

    $ t g4-cmake
    g4-cmake is a function
    g4-cmake ()
    {
        local iwd=$PWD;
        local bdir=$(g4-bdir);
        mkdir -p $bdir;
        local idir=$(g4-prefix);
        mkdir -p $idir;
        g4-bcd;
        cmake -G "$(opticks-cmake-generator)" -DCMAKE_BUILD_TYPE=Debug -DGEANT4_INSTALL_DATA=ON -DGEANT4_USE_GDML=ON -DXERCESC_ROOT_DIR=$(xercesc-prefix) -DCMAKE_INSTALL_PREFIX=$idir $(g4-dir);
        cd_func $iwd
    }

    ntuhep@ntuhep-PC MINGW64 /usr/local/opticks/externals/g4
    $ opticks-cmake-generator
    MSYS Makefiles

::

    [  4%] Linking CXX static library ../../BuildProducts/lib/lib_G4intercoms-archive.a
    [  4%] Built target _G4intercoms-archive
    Scanning dependencies of target genwindef

    [  4%] Building CXX object CMakeFiles/genwindef.dir/cmake/Modules/genwindef/genwindef.cpp.obj
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/genwindef.cpp:6:0: warning: ignoring #pragma warning  [-Wunknown-pragmas]
       #pragma warning ( disable : 4273 )
     ^
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/genwindef.cpp: In function 'int main(int, char**)':
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/genwindef.cpp:58:11: warning: statement has no effect [-Wunused-value]
       for (arg; arg < argc; arg++) {
               ^
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/genwindef.cpp:36:8: warning: unused variable 'debug' [-Wunused-variable]
       bool debug(false);
            ^

    [  4%] Building CXX object CMakeFiles/genwindef.dir/cmake/Modules/genwindef/LibSymbolInfo.cpp.obj
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/LibSymbolInfo.cpp: In member function 'BOOL CLibSymbolInfo::Dump(LPTSTR, std::ostr                             eam&)':
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/LibSymbolInfo.cpp:18:59: error: cast from 'PSTR {aka char*}' to 'DWORD {aka long u                             nsigned int}' loses precision [-fpermissive]
     #define MakePtr( cast, ptr, addValue ) (cast)( (DWORD)(ptr) + (DWORD)(addValue))
                                                               ^
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/LibSymbolInfo.cpp:92:13: note: in expansion of macro 'MakePtr'
       pMbrHdr = MakePtr( PIMAGE_ARCHIVE_MEMBER_HEADER, pArchiveStartString,
                 ^
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/LibSymbolInfo.cpp:18:80: warning: cast to pointer from integer of different size [                             -Wint-to-pointer-cast]
     #define MakePtr( cast, ptr, addValue ) (cast)( (DWORD)(ptr) + (DWORD)(addValue))
                                                                                    ^
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/LibSymbolInfo.cpp:92:13: note: in expansion of macro 'MakePtr'
       pMbrHdr = MakePtr( PIMAGE_ARCHIVE_MEMBER_HEADER, pArchiveStartString,
                 ^
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/LibSymbolInfo.cpp:18:59: error: cast from 'PDWORD {aka long unsigned int*}' to 'DW                             ORD {aka long unsigned int}' loses precision [-fpermissive]
     #define MakePtr( cast, ptr, addValue ) (cast)( (DWORD)(ptr) + (DWORD)(addValue))
                                                               ^
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/LibSymbolInfo.cpp:108:24: note: in expansion of macro 'MakePtr'
       PSTR pszSymbolName = MakePtr( PSTR, pMemberOffsets, 4 * cSymbols );
                            ^
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/LibSymbolInfo.cpp:18:80: warning: cast to pointer from integer of different size [                             -Wint-to-pointer-cast]
     #define MakePtr( cast, ptr, addValue ) (cast)( (DWORD)(ptr) + (DWORD)(addValue))
                                                                                    ^
    C:/msys64/usr/local/opticks/externals/g4/geant4.10.02/cmake/Modules/genwindef/LibSymbolInfo.cpp:108:24: note: in expansion of macro 'MakePtr'
       PSTR pszSymbolName = MakePtr( PSTR, pMemberOffsets, 4 * cSymbols );
                            ^
    CMakeFiles/genwindef.dir/build.make:86: recipe for target 'CMakeFiles/genwindef.dir/cmake/Modules/genwindef/LibSymbolInfo.cpp.obj' failed
    make[2]: *** [CMakeFiles/genwindef.dir/cmake/Modules/genwindef/LibSymbolInfo.cpp.obj] Error 1
    CMakeFiles/Makefile2:437: recipe for target 'CMakeFiles/genwindef.dir/all' failed
    make[1]: *** [CMakeFiles/genwindef.dir/all] Error 2
    Makefile:149: recipe for target 'all' failed
    make: *** [all] Error 2

    ntuhep@ntuhep-PC MINGW64 /usr/local/opticks/externals/g4





EOU
}
g4win-dir(){ echo $(local-base)/env/g4/g4-g4win ; }
g4win-cd(){  cd $(g4win-dir); }
g4win-mate(){ mate $(g4win-dir) ; }
g4win-get(){
   local dir=$(dirname $(g4win-dir)) &&  mkdir -p $dir && cd $dir

}
