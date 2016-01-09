#pragma once

class State ; 

class StateGUI {
    public:
         StateGUI(State* state);
         void gui();
    private:
         State* m_state ; 
};


inline StateGUI::StateGUI(State* state) 
    :
    m_state(state)
{
}




