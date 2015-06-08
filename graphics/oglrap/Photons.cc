#include "Photons.hh"


#ifdef GUI_
#include <imgui.h>
#endif


Photons::Photons(NPY<float>* photons) : PhotonsNPY(photons)
{
}


void Photons::gui()
{
#ifdef GUI_
    if (ImGui::CollapsingHeader("Photon Boundary Selection"))
    {
        gui_boundary_selection();
    }

    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Photon Flag Selection"))
    {
        gui_flag_selection();
    }


#endif
}


void Photons::gui_boundary_selection()
{
#ifdef GUI_
    for(unsigned int i=0 ; i < m_boundaries.size() ; i++)
    {
        std::pair<int, std::string> choice = m_boundaries[i];
        ImGui::Checkbox(choice.second.c_str(), m_boundaries_selection+i );
    }
#endif
}


void Photons::gui_flag_selection()
{
#ifdef GUI_
    for(unsigned int i=0 ; i < m_flags.size() ; i++)
    {
        std::pair<int, std::string> choice = m_flags[i];
        ImGui::Checkbox(choice.second.c_str(), m_flags_selection+i );
    }
#endif
}


