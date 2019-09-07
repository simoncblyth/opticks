/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

// op --gitemindex
#include <cstring>


// npy-
#include "PhotonsNPY.hpp"
#include "Index.hpp"
#include "Types.hpp"

// ggeo-
#include "GItemIndex.hh"


#include "Photons.hh"
#include "OGLRap_imgui.hh"

#include "GUI.hh"


Photons::Photons(Types* types, GItemIndex* boundaries, GItemIndex* seqhis, GItemIndex* seqmat)
    :
    m_types(types),
    m_boundaries(boundaries),
    m_seqhis(seqhis),
    m_seqmat(seqmat)
{
}

const char* Photons::getSeqhisSelectedKey()
{
    return m_seqhis->getSelectedKey();
}
const char* Photons::getSeqhisSelectedLabel()
{
    return m_seqhis ? m_seqhis->getSelectedLabel() : "" ; 
}


void Photons::gui()
{
#ifdef GUI_

    if(m_types)
    {
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Photon Flag Selection"))
        {
            gui_flag_selection();
        }
    }

    if(m_boundaries)
    {
        ImGui::Spacing();
        GUI::gui_radio_select(m_boundaries);
    }

    if(m_seqhis)
    {
        ImGui::Spacing();
        GUI::gui_radio_select(m_seqhis);
    }

    if(m_seqmat)
    {
        ImGui::Spacing();
        GUI::gui_radio_select(m_seqmat);
    }
#endif
}


void Photons::gui_flag_selection()
{
#ifdef GUI_
    // huh not replaced by OpticksAttrSeq ? 
    std::vector<std::string>& labels = m_types->getFlagLabels();  
    bool* flag_selection = m_types->getFlagSelection();
    for(unsigned int i=0 ; i < labels.size() ; i++)
    {
        ImGui::Checkbox(labels[i].c_str(), flag_selection+i );
    }
#endif
}


