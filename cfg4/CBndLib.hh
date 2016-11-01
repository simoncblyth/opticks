#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class OpticksHub ; 
class GBndLib ; 
class GMaterialLib ; 
class GSurfaceLib ; 

class GMaterial ; 
template <typename T> class GPropertyMap ; 

/**
CBndLib
===============

**/

class CFG4_API CBndLib  
{
    public:
        CBndLib(OpticksHub* hub);
        unsigned addBoundary(const char* spec);
    public:
        GMaterial*        getOuterMaterial(unsigned boundary);
        GMaterial*        getInnerMaterial(unsigned boundary);
        GPropertyMap<float>*  getOuterSurface(unsigned boundary);
        GPropertyMap<float>*  getInnerSurface(unsigned boundary);
    private:
        OpticksHub*      m_hub ; 
        GBndLib*         m_bndlib ; 
        GMaterialLib*    m_matlib ; 
        GSurfaceLib*     m_surlib ; 

};


#include "CFG4_TAIL.hh"

