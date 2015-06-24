#pragma once

// adds an optional gui to PhotonsNPY 

class PhotonsNPY ; 
class BoundariesNPY ; 
class Types ; 

class Photons {
   public:
       Photons(PhotonsNPY* photons, BoundariesNPY* boundaries);

       void gui();
       void gui_boundary_selection();
       void gui_flag_selection();

   private:
        PhotonsNPY*       m_photons ;  
        BoundariesNPY*    m_boundaries ;  
        Types*            m_types ; 

};





