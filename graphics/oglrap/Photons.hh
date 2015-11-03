#pragma once
#include <cstddef>

class Types ; 
class GItemIndex ; 

// TODO: Checkbox multiple-select that works with GItemList
//       collapse this class into GUI ?

class Photons {
   public:
       Photons(Types* types, GItemIndex* boundaries, GItemIndex* seqhis, GItemIndex* seqmat);
   public:
       void gui();
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


