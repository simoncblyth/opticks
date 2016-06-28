# === func-gen- : g4/g4win fgp g4/g4win.bash fgn g4win fgh g4
g4win-src(){      echo g4/g4win.bash ; }
g4win-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4win-src)} ; }
g4win-vi(){       vi $(g4win-source) ; }
g4win-env(){      elocal- ; }
g4win-usage(){ cat << EOU

G4 On Windows notes
=====================


Opticks FLAGS
-------------------

/c/usr/local/opticks/build/CMakeCache.txt::

    155 CMAKE_CONFIGURATION_TYPES:STRING=Debug;Release;MinSizeRel;RelWithDebInfo
    156
    157 //Flags used by the compiler during all build types.
    158 CMAKE_CXX_FLAGS:STRING= /DWIN32 /D_WINDOWS /W3 /GR /EHsc
    159
    160 //Flags used by the compiler during debug builds.
    161 CMAKE_CXX_FLAGS_DEBUG:STRING=/D_DEBUG /MDd /Zi /Ob0 /Od /RTC1
    162
    163 //Flags used by the compiler during release builds for minimum
    164 // size.
    165 CMAKE_CXX_FLAGS_MINSIZEREL:STRING=/MD /O1 /Ob1 /D NDEBUG
    166
    167 //Flags used by the compiler during release builds.
    168 CMAKE_CXX_FLAGS_RELEASE:STRING=/MD /O2 /Ob2 /D NDEBUG
    169
    170 //Flags used by the compiler during release builds with debug info.
    171 CMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=/MD /Zi /O2 /Ob1 /D NDEBUG
    172
    173 //Libraries linked by default with all C++ applications.
    174 CMAKE_CXX_STANDARD_LIBRARIES:STRING=kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib
    175


Hmm this doesnt incorporate env/cmake/Modules/EnvCompilationFlags.cmake::

     04 if(WIN32)
      5
      6   # need to detect compiler not os?
      7   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W4")
      8   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
      9   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS")
     10   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS")
     11   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES")
     12

CMake defaults from cmak-;cmak-flags::

    .CMAKE_CXX_FLAGS :  /DWIN32 /D_WINDOWS /W3 /GR /EHsc
    .CMAKE_CXX_FLAGS_DEBUG : /D_DEBUG /MDd /Zi /Ob0 /Od /RTC1
    .CMAKE_CXX_FLAGS_MINSIZEREL : /MD /O1 /Ob1 /D NDEBUG
    .CMAKE_CXX_FLAGS_RELEASE : /MD /O2 /Ob2 /D NDEBUG
    .CMAKE_CXX_FLAGS_RELWITHDEBINFO : /MD /Zi /O2 /Ob1 /D NDEBUG
    .CMAKE_EXE_LINKER_FLAGS :  /machine:X86


G4 MSVC CXX FLAGS
------------------

~/local/opticks/externals/g4/geant4_10_02_p01.build/CMakeCache.txt::


     23 //Semicolon separated list of supported configuration types, only
     24 // supports Debug, Release, MinSizeRel, and RelWithDebInfo, anything
     25 // else will be ignored.
     26 CMAKE_CONFIGURATION_TYPES:STRING=Debug;Release;MinSizeRel;RelWithDebInfo
     27
     28 //Flags used by the compiler during all build types.
     29 CMAKE_CXX_FLAGS:STRING= -GR -EHsc -Zm200 -nologo -D_CONSOLE -D_WIN32 -DWIN32 -DOS -DXPNET -D_CRT_SECURE_NO_DEPRECATE
     30
     31 //Flags used by the compiler during debug builds.
     32 CMAKE_CXX_FLAGS_DEBUG:STRING=-MDd -Od -Zi
     33
     34 //Flags used by the compiler during release builds for minimum
     35 // size.
     36 CMAKE_CXX_FLAGS_MINSIZEREL:STRING=-MD -Os -DNDEBUG
     37
     38 //Flags used by the compiler during release builds.
     39 CMAKE_CXX_FLAGS_RELEASE:STRING=-MD -O2 -DNDEBUG
     40
     41 //Flags used by the compiler during release builds with debug info.
     42 CMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=-MD -O2 -Zi
     43
     44 //Libraries linked by default with all C++ applications.
     45 CMAKE_CXX_STANDARD_LIBRARIES:STRING=kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib
     46


What is in CMakeCache may not be full story::

    ntuhep@ntuhep-PC MINGW64 ~/local/opticks/externals/g4/geant4_10_02_p01
    $ find . -name '*.txt' -exec grep -H CMAKE_CXX_FLAGS {} \;
    ./examples/advanced/brachytherapy/CMakeLists.txt:       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ROOT_CXX_FLAGS}")


    $ find . -name '*.cmake' -exec grep -l CMAKE_CXX_FLAGS {} \;
    ./cmake/Modules/Geant4BuildModes.cmake
    ./cmake/Modules/Geant4BuildProjectConfig.cmake
    ./cmake/Modules/Geant4LibraryBuildOptions.cmake
    ./cmake/Modules/Geant4MakeRules_cxx.cmake
    ./cmake/Templates/UseGeant4.cmake



./cmake/Modules/Geant4MakeRules_cxx.cmake::


     57 # MSVC - all (?) versions
     58 if(MSVC)
     59   # Hmm, WIN32-VC.gmk uses dashes, but cmake uses slashes, latter probably
     60   # best for native build.
     61   set(CMAKE_CXX_FLAGS_INIT "-GR -EHsc -Zm200 -nologo -D_CONSOLE -D_WIN32 -DWIN32 -DOS -DXPNET -D_CRT_SECURE_NO_DEPRECATE")
     62   set(CMAKE_CXX_FLAGS_DEBUG_INIT "-MDd -Od -Zi")
     63   set(CMAKE_CXX_FLAGS_RELEASE_INIT "-MD -O2 -DNDEBUG")
     64   set(CMAKE_CXX_FLAGS_MINSIZEREL_INIT "-MD -Os -DNDEBUG")
     65   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-MD -O2 -Zi")
     66
     67   # Extra modes
     68   set(CMAKE_CXX_FLAGS_TESTRELEASE_INIT "-MDd -Zi -G4DEBUG_VERBOSE")
     69   set(CMAKE_CXX_FLAGS_MAINTAINER_INIT "-MDd -Zi")
     70
     71 endif()


The rules are almost the first thing done in top level CMakeLists.txt::

     23 # - Define CMake requirements and override make rules as needed
     24 #
     25 cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
     26
     27 # - Any policy requirements should go here
     28
     29 set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
     30    ${CMAKE_SOURCE_DIR}/cmake/Modules/Geant4MakeRules_cxx.cmake)


Problem with just adopting these is that it prevents a Debug build from talking to 




g4win-;g4win-configure::

    G4Win.Geant4_INCLUDE_DIRS : C:/Users/ntuhep/local/opticks/externals/include/Geant4;/usr/local/env/windows/ome/xerces-c-3.1.3/src
    G4Win.Geant4_DEFINITIONS : -DG4_STORE_TRAJECTORY;-DG4VERBOSE;-DG4UI_USE;-DG4VIS_USE
    G4Win.Geant4_CXX_FLAGS :  -GR -EHsc -Zm200 -nologo -D_CONSOLE -D_WIN32 -DWIN32 -DOS -DXPNET -D_CRT_SECURE_NO_DEPRECATE -DG4USE_STD11

    // only -DG4USE_STD11 added relative to CMakeCache

    G4Win.Geant4_CXX_FLAGS_DEBUG : -MDd -Od -Zi
    G4Win.Geant4_CXX_FLAGS_MINSIZEREL : -MD -Os -DNDEBUG
    G4Win.Geant4_CXX_FLAGS_RELEASE : -MD -O2 -DNDEBUG
    G4Win.Geant4_CXX_FLAGS_RELWITHDEBINFO : -MD -O2 -Zi
    G4Win.Geant4_EXE_LINKER_FLAGS :  /machine:X86



cmake/Templates/UseGeant4.cmake::


     10 #
     11 #  include(${Geant4_USE_FILE})
     12 #
     13 # results in the addition of the Geant4 compile definitions and
     14 # include directories to those of the directory in which this file is
     15 # included.
     16 #
     17 # Header paths are added to include_directories as SYSTEM type directories
     18 #
     19 # The recommended Geant4 compiler flags are also prepended to
     20 # CMAKE_CXX_FLAGS but duplicated flags are NOT removed. This permits
     21 # client of UseGeant4 to override Geant4's recommended flags if required
     22 # and at their own risk.
     23 #
     24 # Advanced users requiring special sets of flags, or the removal of
     25 # duplicate flags should therefore *not* use this file, preferring the
     26 # direct use of the Geant4_XXXX variables set by the Geant4Config file.
     27 #
     28 # The last thing the module does is to optionally include an internal Use
     29 # file. This file can contain variables, functions and macros for strict
     30 # internal use in Geant4, such as building and running validation tests.
     31 #
     32
     33 #-----------------------------------------------------------------------
     34 # We need to set the compile definitions and include directories
     35 #
     36 add_definitions(${Geant4_DEFINITIONS})
     37 include_directories(AFTER SYSTEM ${Geant4_INCLUDE_DIRS})
     38
     39 #-----------------------------------------------------------------------
     40 # Because Geant4 is sensitive to the compiler flags, let's set the base
     41 # set here. This reproduces as far as possible the behaviour of the
     42 # original makefile system. However, we append any existing CMake flags in
     43 # case the user wishes to override these (at their own risk).
     44 # Though this may lead to duplication, that should not affect behaviour.
     45 #
     46 set(CMAKE_CXX_FLAGS                "${Geant4_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
     47 set(CMAKE_CXX_FLAGS_DEBUG          "${Geant4_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_DEBUG}")
     48 set(CMAKE_CXX_FLAGS_MINSIZEREL     "${Geant4_CXX_FLAGS_MINSIZEREL} ${CMAKE_CXX_FLAGS_MINSIZEREL}")
     49 set(CMAKE_CXX_FLAGS_RELEASE        "${Geant4_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_RELEASE}")
     50 set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${Geant4_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
     51 set(CMAKE_EXE_LINKER_FLAGS         "${Geant4_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
     52







Visual Studio 2015 : Compiler Options
------------------------------------------


* https://msdn.microsoft.com/en-us/library/9s7c9wdw.aspx


GR
    Enables run-time type information (RTTI).
EH
    Specifies the model of exception handling.

EHsc
    If "c" is used with "s" (/EHsc), catches C++ exceptions only 
    and tells the compiler to assume that functions declared 
    as extern "C" never throw a C++ exception.
  
MD
    Creates a multithreaded DLL using MSVCRT.lib.
MDd
    Creates a debug multithreaded DLL using MSVCRTD.lib.

0d
    Disables optimization
O2
    Creates fast code.    
Zi
    Generates complete debugging information.



g4win-;g4win-configure::

    G4Win.Geant4_INCLUDE_DIRS : C:/Users/ntuhep/local/opticks/externals/include/Geant4;/usr/local/env/windows/ome/xerces-c-3.1.3/src
    G4Win.Geant4_DEFINITIONS : -DG4_STORE_TRAJECTORY;-DG4VERBOSE;-DG4UI_USE;-DG4VIS_USE
    G4Win.Geant4_CXX_FLAGS :  -GR -EHsc -Zm200 -nologo -D_CONSOLE -D_WIN32 -DWIN32 -DOS -DXPNET -D_CRT_SECURE_NO_DEPRECATE -DG4USE_STD11

    // only -DG4USE_STD11 added relative to CMakeCache

    G4Win.Geant4_CXX_FLAGS_DEBUG          : -MDd -Od -Zi

    G4Win.Geant4_CXX_FLAGS_RELWITHDEBINFO : -MD  -O2 -Zi
    G4Win.Geant4_CXX_FLAGS_RELEASE        : -MD  -O2      -DNDEBUG
    G4Win.Geant4_CXX_FLAGS_MINSIZEREL     : -MD  -Os      -DNDEBUG

    G4Win.Geant4_EXE_LINKER_FLAGS :  /machine:X86


Visual Studio defines _DEBUG when you specify the /MTd or /MDd option, 
NDEBUG disables standard-C assertions.



NDEBUG
----------

assert Macro

* https://msdn.microsoft.com/en-us/library/9sb57dw4.aspx

The assert macro is enabled in both the release and debug versions of the C
run-time libraries when NDEBUG is not defined. When NDEBUG is defined, the
macro is available but does not evaluate its argument and has no effect.




_ITERATOR_DEBUG_LEVEL
--------------------------

* https://msdn.microsoft.com/en-us/library/hh697468.aspx

The _ITERATOR_DEBUG_LEVEL (IDL) macro supersedes and combines the functionality of the _SECURE_SCL (SCL) 
and _HAS_ITERATOR_DEBUGGING (HID) macros.

::


    Debug    IDL=0               SCL=0, HID=0                  Disables checked iterators and disables iterator debugging.
    Debug    IDL=1               SCL=1, HID=0                  Enables checked iterators and disables iterator debugging.
    Debug    IDL=2 (Default)     SCL=(does not apply), HID=1   By default, enables iterator debugging; checked iterators are not relevant.

    Release  IDL=0 (Default)     SCL=0                         By default, disables checked iterators.
    Release  IDL=1               SCL=1                         Enables checked iterators; iterator debugging is not relevant.


Getting Debug code (-MDd) to talk to Release (-MD) code ?
-----------------------------------------------------------

Seems that setting _ITERATOR_DEBUG_LEVEL to 0  in Debug should allow it to talk with Release binaries ?



Boost complains at link
~~~~~~~~~~~~~~~~~~~~~~~~~

::

     libboost_regex-vc140-mt-gd-1_61.lib(regex_raw_buffer.obj) :
     error LNK2038: mismatch detected for '_ITERATOR_DEBUG_LEVEL': value '2' doesn't match value '0' 
     in BRAP_LOG.obj [C:\usr\local\opticks\build\boostrap\BoostRap.vcxproj]


Move to using the Release static Boost libs still complaint::

     libboost_regex-vc140-mt-1_61.lib(w32_regex_traits.obj) : 
     error LNK2038: mismatch detected for 'RuntimeLibrary': value 'MD_DynamicRelease' doesn't match value 'MDd_DynamicDebug' 
     in BRAP_LOG.obj [C:\usr\local\opticks\build\boostrap\BoostRap.vcxproj]


Nope so try using RelWithDebInfo for Opticks.
Using opticks-cleanbuild to wipe everything (other than externals) when making such changes.

This gets further... 






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


g4win-dir(){    echo $(env-home)/g4 ; }
g4win-sdir(){   echo $(local-base)/env/g4win/g4win.source ; }
g4win-bdir(){   echo $(local-base)/env/g4win/g4win.build ; }
g4win-prefix(){ echo $(local-base)/env/g4win/g4win.install ; }

g4win-cd(){    cd $(g4win-dir); }
g4win-scd(){   cd $(g4win-sdir); }
g4win-bcd(){   cd $(g4win-bdir); }
g4win-icd(){   cd $(g4win-idir); }

g4win-config(){ echo Debug ; }

g4win-cmake(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD
   local bdir=$(g4win-bdir)
   mkdir -p $bdir
   [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already use g4win-configure to reconfigure  && return

   g4win-bcd

   g4-
   xercesc-

   cmake \
       -DCMAKE_INSTALL_PREFIX=$(g4win-prefix) \
       -DGeant4_DIR=$(g4-cmake-dir) \
       -DXERCESC_LIBRARY=$(xercesc-library) \
       -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
       $* \
       $(g4win-sdir)

   cd $iwd
}

g4win-wipe(){
   local bdir=$(g4win-bdir)
   rm -rf $bdir
}

g4win-configure(){

   local sdir=$(g4win-sdir)
   [ ! -d "$sdir" ] && mkdir -p $sdir

   g4win-scd

   # do everything, everytime
  
   rm -f CMakeLists.txt
   g4win-cmak-txt- > CMakeLists.txt
   g4win-wipe 
 
   g4win-cmake 
}



g4win--(){

   local msg="$FUNCNAME : "
   local iwd=$PWD

   local bdir=$(g4win-bdir)
   [ ! -d "$bdir" ] && echo $msg bdir $bdir does not exist && return

   cd $bdir

   cmake --build . --config $(g4win-config) --target ${1:-install}

   cd $iwd
}


g4win-cmak-vars-(){ local name=${1:-Geant4} ; cat << EOV
${name}_LIBRARY
${name}_LIBRARIES
${name}_INCLUDE_DIRS
${name}_DEFINITIONS
${name}_CXX_FLAGS
${name}_CXX_FLAGS_DEBUG
${name}_CXX_FLAGS_MINSIZEREL
${name}_CXX_FLAGS_RELEASE
${name}_CXX_FLAGS_RELWITHDEBINFO
${name}_EXE_LINKER_FLAGS
EOV
}


g4win-cmak-txt-(){
     local name=$1
     cat << EOH
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(name G4Win)
project(\${name})

find_package(Geant4 REQUIRED)

EOH
     local var 
     g4win-cmak-vars- | while read var ; do 
          cat << EOM   
message("\${name}.$var : \${$var} ")
EOM
     done
}




