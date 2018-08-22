#pragma once

template <typename T> class NPY ;
#include "G4Types.hh"
#include "CFG4_API_EXPORT.hh"

/**
CPhotonCollector
===================

Photons (item shape 4*4, 4 quads)
-------------------------------------

**/

class CFG4_API CPhotonCollector 
{
    public:
        static CPhotonCollector* Instance();
    public:
        CPhotonCollector();  

        std::string description() const ;
        void Summary(const char* msg="CPhotonCollector::Summary") const  ;
    public:
        NPY<float>*  getPhoton() const ;
        void save(const char* path) const ; 
    public:
        void collectPhoton(
               G4double  pos_x,
               G4double  pos_y,
               G4double  pos_z,
               G4double  time,

               G4double  dir_x,
               G4double  dir_y,
               G4double  dir_z,
               G4double  weight,

               G4double  pol_x,
               G4double  pol_y,
               G4double  pol_z,
               G4double  wavelength,

               int flags_x,
               int flags_y,
               int flags_z,
               int flags_w
        );
    private:
        static CPhotonCollector* INSTANCE ;      
        NPY<float>*  m_photon ;
        unsigned     m_photon_itemsize ; 
        float*       m_photon_values ;  
        unsigned     m_photon_count ;  

};



