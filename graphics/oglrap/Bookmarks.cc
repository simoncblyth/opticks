// oglrap-
#include "Bookmarks.hh"
#include "Interactor.hh"

#include <cstring>

#ifdef GUI_
#include <imgui.h>
#endif


#include <boost/lexical_cast.hpp>

// npy-
#include "dirutil.hpp"
#include "NState.hpp"
#include "NLog.hpp"


Bookmarks::Bookmarks(NState* state)  
       :
       m_state(state),
       m_current(0),
       m_current_gui(0)
{
    init();
}

void Bookmarks::init()
{
    typedef std::vector<std::string> VS ;
    VS basenames ; 
    dirlist(basenames, m_state->getDir(), ".ini" );  // basenames do not include the .ini

    for(VS::const_iterator it=basenames.begin() ; it != basenames.end() ; it++)
    {
        std::string basename = *it ; 
        int num(-1);
        try
        { 
            num = boost::lexical_cast<int>(basename) ;
        }   
        catch (const boost::bad_lexical_cast& e ) 
        { 
            LOG(warning)  << "Caught bad lexical cast with error " << e.what() ;
        }   
        catch( ... )
        {
            LOG(warning) << "Unknown exception caught!" ;
        }   
        if(num == -1) continue ; 

        m_bookmarks[num] = basename ; 
        LOG(info) << "Bookmarks::init " << num ;  
    }
}


void Bookmarks::number_key_pressed(unsigned int num, unsigned int modifiers)
{
    LOG(info) << "Bookmarks::number_key_pressed "
              << " num "  << num 
              << " modifiers " << Interactor::describeModifiers(modifiers) 
              ; 

    if(!exists(num))
    {
        add(num);    
    }
    else
    {
        setCurrent(num);
        apply();
    }
}

void Bookmarks::number_key_released(unsigned int num)
{
    LOG(info) << "Bookmarks::number_key_released " << num ; 
}


void Bookmarks::add(unsigned int num)
{
    setCurrent(num);
    collect();
}

void Bookmarks::collect()
{
    if(m_current == 0 ) return ; 

    m_state->collect();
    m_state->setName(m_current);
    m_state->save();
}

void Bookmarks::apply()
{
    if(m_current == 0 ) return ; 

    m_state->setName(m_current);
    m_state->load();
    m_state->apply();
}


void Bookmarks::gui()
{
#ifdef GUI_
    ImGui::SameLine();
    if(ImGui::Button("collect")) collect();
    ImGui::SameLine();
    if(ImGui::Button("apply")) apply();


    typedef std::map<unsigned int, std::string>::const_iterator MUSI ; 

    for(MUSI it=m_bookmarks.begin() ; it!=m_bookmarks.end() ; it++)
    {
         unsigned int num = it->first ; 
         std::string name = it->second ; 
         ImGui::RadioButton(name.c_str(), &m_current_gui, num);
    }


    // not directly setting m_current as need to notice a change
    if(m_current_gui != m_current ) 
    {
        setCurrent(m_current_gui);
        ImGui::Text(" changed : %d ", m_current);
        apply();
    }
#endif
}


