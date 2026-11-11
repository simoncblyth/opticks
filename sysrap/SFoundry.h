#pragma once
/**
SFoundry.h
===========

This is a pure virtual base of CSGFoundry - it encapsulates
the subset of the CSGFoundry API needed to create the OptiX geometry.

TODO: see if any trims to this API are possible - see thoughts at head of CSG/CSGFoundry.cc

**/

#include <string>

struct SSim ;
struct SGLM ;
struct SScene ;
struct SDeviceGeometry ;
struct SCSGPrimSpec ;

struct CSGSolid ;
struct CSGPrim ;

struct SFoundry
{
    virtual void upload()                                              = 0 ;
    virtual const SDeviceGeometry* getDeviceGeometry() const             = 0 ;

    virtual std::string descBase() const                               = 0 ;
    virtual const SSim* getSim() const                                 = 0 ;
    virtual void exportTreeSceneFrame( SGLM* gm ) const                = 0 ;
    virtual SScene* getScene() const                                   = 0 ;

    virtual unsigned getNumSolid() const                               = 0 ;
    virtual unsigned getNumPrim() const                                = 0 ;

    virtual bool              isSolidTrimesh(int gas_idx) const            = 0 ;
    virtual const CSGSolid*   getSolid(unsigned solidIdx) const            = 0 ;
    virtual SCSGPrimSpec getPrimSpec( unsigned solidIdx) const             = 0 ;
    virtual const std::string& getSolidMMLabel(unsigned solidIdx) const    = 0 ;

    virtual const CSGPrim*    getPrim(unsigned primIdx) const              = 0 ;
    virtual int               getPrimBoundary_( const CSGPrim* pr ) const  = 0 ;

    virtual unsigned getNumUniqueIAS() const                               = 0 ;
    virtual unsigned getNumInst() const                                    = 0 ;
    virtual unsigned getNumInstancesIAS(int ias_idx, unsigned long long emm) const = 0 ;
    virtual void     getInstanceTransformsIAS(std::vector<qat4>& select_inst, int ias_idx, unsigned long long emm ) const = 0 ;

    virtual const char* getOriginCFBase() const                        = 0 ;
    virtual const std::string& getMeta() const                         = 0 ;

};


