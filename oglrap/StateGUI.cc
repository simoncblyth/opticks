#include "StateGUI.hh"
#include "NState.hpp"
#include "PLOG.hh"

#ifdef GUI_
#include <ImGui/imgui.h>
#endif


StateGUI::StateGUI(NState* state) 
    :
    m_state(state)
{
}


void StateGUI::gui()
{
#ifdef GUI_

    if(ImGui::Button("roundtrip")) m_state->roundtrip();
    ImGui::SameLine();
    if(ImGui::Button("save")) m_state->save();
    ImGui::SameLine();
    if(ImGui::Button("load")) m_state->load();
    ImGui::SameLine();

    if (ImGui::CollapsingHeader("StateString"))
    {
        ImGui::Text("%s",m_state->getStateString().c_str());
        if(ImGui::Button("update")) m_state->update();
    }



#endif
}




