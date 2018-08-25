#pragma once

class NGunConfig ; 

#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"

/**
CGunSource
============

Converts NGunConfig into G4VPrimaryGenerator 
with GeneratePrimaryVertex(G4Event *evt)

**/


class CFG4_API CGunSource: public CSource
{
    public:
        CGunSource(Opticks* ok);
        virtual ~CGunSource();
        void configure(NGunConfig* gc);
    public:
        void GeneratePrimaryVertex(G4Event* event);
    private:
        NGunConfig*   m_config ; 

};


