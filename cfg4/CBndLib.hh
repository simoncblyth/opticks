#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class OpticksHub ; 
class GBndLib ; 
class GMaterialLib ; 
class GSurfaceLib ; 

class GMaterial ; 

//class GSurLib ; 
//class GSur ; 

/**
CBndLib
===============

Q: why ? 

Eliminate this class : it does too little

**/

class CFG4_API CBndLib  
{
    public:
        CBndLib(OpticksHub* hub);
        unsigned addBoundary(const char* spec);
    public:
        GMaterial*        getOuterMaterial(unsigned boundary);
        GMaterial*        getInnerMaterial(unsigned boundary);
        GPropertyMap<float>* getOuterSurface(unsigned boundary);
        GPropertyMap<float>* getInnerSurface(unsigned boundary);

        //GSur*             getOuterSurface(unsigned boundary);
        //GSur*             getInnerSurface(unsigned boundary);
    private:
        OpticksHub*      m_hub ; 
        GBndLib*         m_blib ; 
        GMaterialLib*    m_mlib ; 
        GSurfaceLib*     m_slib ; 
        //GSurLib*         m_slib ; 

};


#include "CFG4_TAIL.hh"

