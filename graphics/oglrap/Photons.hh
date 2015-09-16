#pragma once

// adds an optional gui to PhotonsNPY 
#include "stdlib.h"

class Types ; 
class PhotonsNPY ; 
class BoundariesNPY ; 
class Types ; 
class GItemIndex ; 

class Photons {
   public:
       Photons(Types* types, PhotonsNPY* photons, BoundariesNPY* boundaries, GItemIndex* seqhis, GItemIndex* seqmat);
   public:
       void gui();
       void gui_boundary_selection();
       void gui_flag_selection();
   public:
        Types*         getTypes();
        PhotonsNPY*    getPhotons();
        BoundariesNPY* getBoundaries();
        GItemIndex*    getSeqHis();
        GItemIndex*    getSeqMat();
   private:
        Types*            m_types ; 
        PhotonsNPY*       m_photons ;  
        BoundariesNPY*    m_boundaries ;  
        GItemIndex*       m_seqhis ;  
        GItemIndex*       m_seqmat ;  

};


inline Photons::Photons(Types* types, PhotonsNPY* photons, BoundariesNPY* boundaries, GItemIndex* seqhis, GItemIndex* seqmat)
    :
    m_types(types),
    m_photons(photons),
    m_boundaries(boundaries),
    m_seqhis(seqhis),
    m_seqmat(seqmat)
{
}


inline Types* Photons::getTypes()
{
    return m_types ; 
}
inline PhotonsNPY* Photons::getPhotons()
{
    return m_photons ; 
}
inline BoundariesNPY* Photons::getBoundaries()
{
    return m_boundaries ; 
}
inline GItemIndex* Photons::getSeqHis()
{
    return m_seqhis ; 
}
inline GItemIndex* Photons::getSeqMat()
{
    return m_seqmat ; 
}

