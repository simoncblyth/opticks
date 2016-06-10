# === func-gen- : g4/g4win fgp g4/g4win.bash fgn g4win fgh g4
g4win-src(){      echo g4/g4win.bash ; }
g4win-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4win-src)} ; }
g4win-vi(){       vi $(g4win-source) ; }
g4win-env(){      elocal- ; }
g4win-usage(){ cat << EOU

G4 On Windows notes
=====================



MSVC Approach
----------------

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/ch01.html

Officially supported: Windows 7 with Visual Studio 2013 or 2015, 64bit.


* web/installer download of Visual Studio 2015 community 



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
