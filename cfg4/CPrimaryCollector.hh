#pragma once

template <typename T> class NPY ;
#include "G4Types.hh"
#include "CFG4_API_EXPORT.hh"

/**
CPrimaryCollector
===================

Primaries (item shape 4*4, 4 quads)
-------------------------------------

Primary collection is invoked from CSource::collectPrimary(G4PrimaryVertex* vtx)
into the CPrimaryCollector singleton instance.

**/

class CFG4_API CPrimaryCollector 
{
    public:
        static CPrimaryCollector* Instance();
    private:
        static CPrimaryCollector* INSTANCE ;      
    public:
        CPrimaryCollector();  
        std::string description() const ;
        void Summary(const char* msg="CPrimaryCollector::Summary") const  ;
    public:
        NPY<float>*  getPrimary() const ;
    public:
        void collectPrimary(
               G4double  x0,
               G4double  y0,
               G4double  z0,
               G4double  t0,

               G4double  dir_x,
               G4double  dir_y,
               G4double  dir_z,
               G4double  weight,

               G4double  pol_x,
               G4double  pol_y,
               G4double  pol_z,
               G4double  wavelength,

               unsigned flags_x,
               unsigned flags_y,
               unsigned flags_z,
               unsigned flags_w
        );
    private:
        NPY<float>*  m_primary ;
        unsigned     m_primary_itemsize ; 
        float*       m_primary_values ;  
        unsigned     m_primary_count ;  

};



