#include "Photons.hh"
#include "PhotonsNPY.hpp"
#include "BoundariesNPY.hpp"

#ifdef GUI_
#include <imgui.h>
#endif


Photons::Photons(PhotonsNPY* photons, BoundariesNPY* boundaries)
    :
    m_photons(photons),
    m_boundaries(boundaries),
    m_types(photons->getTypes())
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

    std::vector< std::pair<int, std::string> >&  boundaries = m_boundaries->getBoundaries();
    bool* boundaries_selection = m_boundaries->getBoundariesSelection();

    for(unsigned int i=0 ; i < boundaries.size() ; i++)
    {
        std::pair<int, std::string> choice = boundaries[i];
        ImGui::Checkbox(choice.second.c_str(), boundaries_selection+i );
    }
#endif
}


void Photons::gui_flag_selection()
{
#ifdef GUI_
    std::vector<std::string>& labels = m_types->getFlagLabels();  
    bool* flag_selection = m_types->getFlagSelection();
    for(unsigned int i=0 ; i < labels.size() ; i++)
    {
        ImGui::Checkbox(labels[i].c_str(), flag_selection+i );
    }
#endif
}


