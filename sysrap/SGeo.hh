#pragma once

/**
SGeo : protocol fulfilled by CSGFoundry 
=========================================

Protocol base used to facilitate lower level package access
to limited geometry information, by passing the higher level 
CSGFoundry instance down to it cast down to this SGeo protocol base.

Only a little usage from active code::

    epsilon:opticks blyth$ opticks-f SGeo.hh 
    ./CSG/CSGFoundry.h:#include "SGeo.hh"
    ./sysrap/CMakeLists.txt:    SGeo.hh
    ./sysrap/CheckGeo.hh:#include "SGeo.hh"
    ./sysrap/SGeo.hh:    epsilon:sysrap blyth$ opticks-f SGeo.hh 
    ./sysrap/SGeo.hh:    ./CSG/CSGFoundry.h:#include "SGeo.hh"
    ./sysrap/SEvt.cc:#include "SGeo.hh"
    ./sysrap/SGeo.cc:#include "SGeo.hh"

    ##DEAD
    ./GeoChain/GeoChain.cc:#include "SGeo.hh"
    ./ggeo/GGeo.hh:#include "SGeo.hh"
    ./ggeo/tests/GGeoTest.cc:#include "SGeo.hh"
    ./optickscore/OpticksDbg.cc:#include "SGeo.hh"


**/

#include "plog/Severity.h"
#include <string>
#include "SYSRAP_API_EXPORT.hh"
#include "sframe.h"   // Zepeng reports needed for Rocky Linux 9/gcc 11.3 
struct stree ; 

struct SYSRAP_API SGeo 
{
    public:
        static SGeo* Get() ; 
    private:
        static SGeo* INSTANCE ; 
        static const plog::Severity LEVEL ; 
    public:
        static std::string Desc() ; 
    public:
        SGeo(); 
    public:
        virtual unsigned           getNumMeshes() const = 0 ; 
        virtual const char*        getMeshName(unsigned midx) const = 0 ;
        virtual int                getMeshIndexWithName(const char* name, bool startswith) const = 0 ;
        virtual int                getFrame(sframe& fr, int ins_idx ) const = 0 ;
        virtual std::string        descBase() const = 0 ; 
        virtual int                lookup_mtline(int mtindex) const = 0 ; 
        virtual std::string        desc_mt() const = 0 ; 
        virtual stree*             getTree() const = 0 ; 

        virtual ~SGeo(){};

};


