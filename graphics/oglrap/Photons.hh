#pragma once
#include <cstddef>

class Types ; 
class GItemIndex ; 

class Photons {
   public:
       Photons(Types* types, GItemIndex* boundaries, GItemIndex* seqhis, GItemIndex* seqmat);
   public:
       void gui();
       //void gui_boundary_selection();
       void gui_flag_selection();
       void gui_radio_select(GItemIndex* ii);
       void gui_item_index(GItemIndex* ii);
   private:
        Types*            m_types ; 
        GItemIndex*       m_boundaries ;  
        GItemIndex*       m_seqhis ;  
        GItemIndex*       m_seqmat ;  

};


inline Photons::Photons(Types* types, GItemIndex* boundaries, GItemIndex* seqhis, GItemIndex* seqmat)
    :
    m_types(types),
    m_boundaries(boundaries),
    m_seqhis(seqhis),
    m_seqmat(seqmat)
{
}


