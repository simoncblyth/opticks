#pragma once

// adds an optional gui to PhotonsNPY 
#include "stdlib.h"

class PhotonsNPY ; 
class BoundariesNPY ; 
class Types ; 
class Index ; 

class Photons {
   public:
       Photons(PhotonsNPY* photons, BoundariesNPY* boundaries, Index* seqhis, Index* seqmat);

       void gui();
       void gui_boundary_selection();
       void gui_flag_selection();
       void gui_radio(Index* index);
   private:  
       void init();
   private:
        PhotonsNPY*       m_photons ;  
        BoundariesNPY*    m_boundaries ;  
        Index*            m_seqhis ;  
        Index*            m_seqmat ;  
        Types*            m_types ; 

};


inline Photons::Photons(PhotonsNPY* photons, BoundariesNPY* boundaries, Index* seqhis, Index* seqmat)
    :
    m_photons(photons),
    m_boundaries(boundaries),
    m_seqhis(seqhis),
    m_seqmat(seqmat),
    m_types(NULL)
{
    init();
}



