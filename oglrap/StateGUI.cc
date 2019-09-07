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

#include "StateGUI.hh"
#include "NState.hpp"
#include "PLOG.hh"

#include "OGLRap_imgui.hh"


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




