#pragma once

class NState ; 

#include "OGLRAP_API_EXPORT.hh"

class OGLRAP_API StateGUI {
    public:
         StateGUI(NState* state);
         void gui();
    private:
         NState*           m_state ; 
};




