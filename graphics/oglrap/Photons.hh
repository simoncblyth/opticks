#pragma once

// adds an optional gui to PhotonsNPY 

class PhotonsNPY ; 
class Types ; 

class Photons {
   public:
       Photons(PhotonsNPY* photons);

       void gui();
       void gui_boundary_selection();
       void gui_flag_selection();

   private:
        PhotonsNPY*    m_photons ;  
        Types*         m_types ; 

};





