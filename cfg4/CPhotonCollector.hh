#pragma once

template <typename T> class NPY ;
#include "CFG4_API_EXPORT.hh"

/**
CPhotonCollector
===================

NB : **No Geant4 dependency** use C4PhotonCollector for that 

This only depends on NPY, so it can be relocated downwards 
to a future intermediary subproj above NPY but below G4 specifics.


Photons (item shape 4*4, 4 quads)
-------------------------------------

**/

class CFG4_API CPhotonCollector 
{
    public:
        static CPhotonCollector* Instance();
    public:
        CPhotonCollector();  

        std::string desc() const ;
        void Summary(const char* msg="CPhotonCollector::Summary") const  ;
    public:
        NPY<float>*  getPhoton() const ;
        void save(const char* path) const ; 
    public:
        void collectPhoton(
               double  pos_x,
               double  pos_y,
               double  pos_z,
               double  time,

               double  dir_x,
               double  dir_y,
               double  dir_z,
               double  weight,

               double  pol_x,
               double  pol_y,
               double  pol_z,
               double  wavelength,

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



