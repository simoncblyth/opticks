#pragma once

/**
SGeo
======

Protocol base used to facilitate lower level package access
to limited geometry information, by passing the higher level 
GGeo instance down to it cast down to this SGeo protocol base.

Also used by CSG/CSGFoundry::upload to record the CFBase directory 
of the last geometry uploaded to the device in a location that
is accessible from anywhere. (HMM: an alt approach would be to set an envvar for this ?)

**/

#include "plog/Severity.h"
#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct sframe ; 

struct SYSRAP_API SGeo 
{
    private:
        static const char* LAST_UPLOAD_CFBASE ;
        static const plog::Severity LEVEL ; 
    public:
        static void SetLastUploadCFBase(const char* cfbase);   
        static const char* LastUploadCFBase() ; 
        static const char* LastUploadCFBase_OutDir(); 
    public:
        virtual unsigned           getNumMeshes() const = 0 ; 
        virtual const char*        getMeshName(unsigned midx) const = 0 ;
        virtual int                getMeshIndexWithName(const char* name, bool startswith) const = 0 ;
        virtual int                getFrame(sframe& fr, int ins_idx ) const = 0 ;
        virtual std::string        descBase() const = 0 ; 

        virtual ~SGeo(){};

};


