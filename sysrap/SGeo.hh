#pragma once

/**
SGeo : protocol fulfilled by CSGFoundry
=========================================

Protocol base used to facilitate lower level package access
to limited geometry information, by passing the higher level
CSGFoundry instance down to it cast down to this SGeo protocol base.

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


    public:
        // collect CSGFoundry API needed by CSGOptiX
        virtual void               upload() = 0 ;



        virtual ~SGeo(){};

};


