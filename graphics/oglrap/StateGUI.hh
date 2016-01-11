#pragma once

class NState ; 

class StateGUI {
    public:
         StateGUI(NState* state);
         void gui();
    private:
         NState*           m_state ; 
};


inline StateGUI::StateGUI(NState* state) 
    :
    m_state(state)
{
}




