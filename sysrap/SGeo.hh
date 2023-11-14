#pragma once

/**
SGeo : protocol fulfilled by CSGFoundry 
=========================================

Protocol base used to facilitate lower level package access
to limited geometry information, by passing the higher level 
CSGFoundry instance down to it cast down to this SGeo protocol base.

Also used by CSG/CSGFoundry::upload to record the CFBase directory 
of the last geometry uploaded to the device in a location that
is accessible from anywhere. (HMM: an alt approach would be to set an envvar for this ?)


::

    epsilon:sysrap blyth$ opticks-f SGeo.hh 
    ./CSG/CSGFoundry.h:#include "SGeo.hh"

    ./sysrap/CMakeLists.txt:    SGeo.hh
    ./sysrap/CheckGeo.hh:#include "SGeo.hh"
    ./sysrap/tests/SGeoTest.cc:#include "SGeo.hh"
    ./sysrap/SEvt.cc:#include "SGeo.hh"
    ./sysrap/SGeo.cc:#include "SGeo.hh"

    ./g4cx/G4CXOpticks.cc:#include "SGeo.hh"


**/

#include "plog/Severity.h"
#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct sframe ; 

struct SYSRAP_API SGeo 
{
    public:
        static SGeo* Get() ; 
    private:
        static SGeo* INSTANCE ; 
        //static const char* LAST_UPLOAD_CFBASE ;
        static const plog::Severity LEVEL ; 
    public:
        /*
        static void SetLastUploadCFBase(const char* cfbase);   
        static const char* LastUploadCFBase() ; 
        static const char* LastUploadCFBase_OutDir(); 
        static const char* DefaultDir() ; 
        */

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

        virtual ~SGeo(){};

};


