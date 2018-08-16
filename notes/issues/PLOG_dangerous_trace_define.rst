PLOG_dangerous_trace_define
=============================

Dangerous PLOG.hh trace defines are biting in examples/Geant4/GDMLMangledLVNames with xercesc headers.
Have to put the OPTICKS_LOG.hh include last to avoid, 
Trying to eliminate the define of trace.

::


   grep LOG\(trace\) *.*
   perl -pi -e 's,LOG\(trace\),LOG(verbose),g' *.*

::

    Scanning dependencies of target GDMLMangledLVNames
    [ 33%] Building CXX object CMakeFiles/GDMLMangledLVNames.dir/GDMLMangledLVNames.cc.o
    In file included from /Users/blyth/opticks/examples/Geant4/GDMLMangledLVNames/GDMLMangledLVNames.cc:7:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLParser.hh:43:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadStructure.hh:46:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadParamvol.hh:43:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadSetup.hh:47:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadSolids.hh:44:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadMaterials.hh:45:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadDefine.hh:45:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLRead.hh:42:
    In file included from /opt/local/include/xercesc/parsers/XercesDOMParser.hpp:26:
    In file included from /opt/local/include/xercesc/parsers/AbstractDOMParser.hpp:26:
    In file included from /opt/local/include/xercesc/framework/XMLDocumentHandler.hpp:27:
    In file included from /opt/local/include/xercesc/framework/XMLAttr.hpp:26:
    In file included from /opt/local/include/xercesc/util/QName.hpp:30:
    In file included from /opt/local/include/xercesc/internal/XSerializable.hpp:25:
    /opt/local/include/xercesc/internal/XSerializeEngine.hpp:543:27: error: non-friend class member 'verbose' cannot have a qualified name
        void                  trace(char*)     const;
                              ^~~~~
    /usr/local/opticks/include/SysRap/PLOG.hh:18:21: note: expanded from macro 'trace'
    #define trace plog::verbose 
                  ~~~~~~^
    In file included from /Users/blyth/opticks/examples/Geant4/GDMLMangledLVNames/GDMLMangledLVNames.cc:7:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLParser.hh:43:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadStructure.hh:46:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadParamvol.hh:43:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadSetup.hh:47:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadSolids.hh:44:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadMaterials.hh:45:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLReadDefine.hh:45:
    In file included from /usr/local/opticks/externals/include/Geant4/G4GDMLRead.hh:42:
    In file included from /opt/local/include/xercesc/parsers/XercesDOMParser.hpp:26:
    In file included from /opt/local/include/xercesc/parsers/AbstractDOMParser.hpp:26:
    In file included from /opt/local/include/xercesc/framework/XMLDocumentHandler.hpp:27:
    In file included from /opt/local/include/xercesc/framework/XMLAttr.hpp:26:
    In file included from /opt/local/include/xercesc/util/QName.hpp:30:
    In file included from /opt/local/include/xercesc/internal/XSerializable.hpp:25:
    /opt/local/include/xercesc/internal/XSerializeEngine.hpp:543:44: error: non-member function cannot have 'const' qualifier
        void                  trace(char*)     const;
                                               ^
    /Users/blyth/opticks/examples/Geant4/GDMLMangledLVNames/GDMLMangledLVNames.cc:30:23: error: use of undeclared identifier 'world'
        gdml->Write(path, world, refs, schemaLocation );
                          ^
    3 errors generated.



