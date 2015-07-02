#include "Photons.hh"


// npy-
#include "PhotonsNPY.hpp"
#include "BoundariesNPY.hpp"
#include "Index.hpp"


#ifdef GUI_
#include <imgui.h>
#endif


void Photons::init()
{
    m_types = m_photons->getTypes() ;
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

    if(m_seqhis)
    {
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Photon Flag Sequence Selection"))
        {
            gui_radio(m_seqhis);
        }
    }

    if(m_seqmat)
    {
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Photon Material Sequence Selection"))
        {
            gui_radio(m_seqmat);
        }
    }
#endif
}



void Photons::gui_radio(Index* index)
{
#ifdef GUI_
   typedef std::vector<std::string> VS ; 
   VS names = index->getNames();
   int* ptr = index->getSelectedPtr();
   ImGui::RadioButton( "All", ptr, 0 );
   for(VS::iterator it=names.begin() ; it != names.end() ; it++ )
   {   
       std::string iname = *it ; 
       unsigned int local  = index->getIndexLocal(iname.c_str()) ;
       ImGui::RadioButton( iname.c_str(), ptr, local);
   }
   ImGui::Text("%s %d ", index->getItemType(), *ptr);
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


