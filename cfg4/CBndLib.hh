#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class OpticksHub ; 
class GBndLib ; 
class GMaterialLib ; 
class GSurLib ; 

class GMaterial ; 
class GSur ; 

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
        GSur*             getOuterSurface(unsigned boundary);
        GSur*             getInnerSurface(unsigned boundary);
    private:
        OpticksHub*      m_hub ; 
        GBndLib*         m_bndlib ; 
        GMaterialLib*    m_matlib ; 
        GSurLib*         m_surlib ; 

};


#include "CFG4_TAIL.hh"

