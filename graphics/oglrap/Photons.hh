#pragma once

#include "PhotonsNPY.hpp"

// adds an optional gui to PhotonsNPY 

class Photons : public PhotonsNPY {
   public:
       Photons(NPY<float>* photons);

       void gui();
       void gui_boundary_selection();
       void gui_flag_selection();

};
