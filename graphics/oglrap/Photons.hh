#pragma once

// adds an optional gui to PhotonsNPY 

class PhotonsNPY ; 
class BoundariesNPY ; 
class SequenceNPY ; 
class Types ; 
class Index ; 

class Photons {
   public:
       Photons(PhotonsNPY* photons, BoundariesNPY* boundaries, SequenceNPY* sequence);

       void gui();
       void gui_boundary_selection();
       void gui_flag_selection();
       void gui_radio(Index* index);

   private:
        PhotonsNPY*       m_photons ;  
        BoundariesNPY*    m_boundaries ;  
        SequenceNPY*      m_sequence ;  
        Types*            m_types ; 

};





