multi_launch_forces_metadata_rejig
=====================================

::

    P[blyth@localhost tests]$ opticks-f descMetaKVS
    ./sysrap/NP.hh:    std::string descMetaKVS(const char* juncture=nullptr, const char* ranges=nullptr) const ; 
    ./sysrap/NP.hh:Essentially this is just a selected key version of the full descMetaKVS
    ./sysrap/NP.hh:inline std::string NP::descMetaKVS(const char* juncture_, const char* ranges_) const 
    ./sysrap/NP.hh:    ss << "[NP::descMetaKVS\n" ; 
    ./sysrap/NP.hh:       << "]NP::descMetaKVS\n" 
    ./sysrap/NPFold.h:    std::string descMetaKVS() const ; 
    ./sysrap/NPFold.h:inline std::string NPFold::descMetaKVS() const
    ./sysrap/NPFold.h:    ss << descMetaKVS() << std::endl ; 
    ./sysrap/tests/sreport.cc:       << "[sreport.desc_run.descMetaKVS " << std::endl 
    ./sysrap/tests/sreport.cc:       << ( run ? run->descMetaKVS(JUNCTURE, RANGES) : "-" ) << std::endl
    ./sysrap/tests/sreport.cc:       << "]sreport.desc_run.descMetaKVS " << std::endl 
    ./sysrap/tests/sreport.cc:       << ".sreport_Creator.desc_run.descMetaKVS " << std::endl 
    ./sysrap/tests/sreport.cc:       << ( run ? run->descMetaKVS() : "-" ) << std::endl
    P[blyth@localhost opticks]$ 




    NPFold.h : NP.hh NPX.h

    NPX.h : NP.hh

    NP.hh : NPU.hh

    NPU.hh : system headers only






 

