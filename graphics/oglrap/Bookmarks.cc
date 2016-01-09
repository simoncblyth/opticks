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
       m_current(UNSET),
       m_current_gui(UNSET),
       m_verbose(false)
{
    init();
}

void Bookmarks::init()
{
    readdir();
}


void Bookmarks::updateTitle()
{
    m_title[N] = '\0' ;
    for(unsigned int i=0 ; i < N ; i++) m_title[i] = exists(i) ? i + '0' : '_' ; 
}


void Bookmarks::readdir()
{
    m_bookmarks.clear();

    typedef std::vector<std::string> VS ;
    VS basenames ; 
    dirlist(basenames, m_state->getDir(), ".ini" );  // basenames do not include the .ini

    if(m_verbose)
    LOG(info) << "Bookmarks::readdir " << m_state->getDir() ;

    for(VS::const_iterator it=basenames.begin() ; it != basenames.end() ; it++)
    {
        std::string basename = *it ; 

        if(m_verbose)
        LOG(info) << "Bookmarks::readdir basename " << basename ; 

        int num(UNSET);
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
        if(num == UNSET) continue ; 

        m_bookmarks[num] = basename ; 

        if(m_verbose) 
        LOG(info) << "Bookmarks::readdir " << num ;  
    }


    updateTitle();
}

void Bookmarks::number_key_released(unsigned int num)
{
    LOG(debug) << "Bookmarks::number_key_released " << num ; 
}

void Bookmarks::number_key_pressed(unsigned int num, unsigned int modifiers)
{
    bool exists_ = exists(num);
    if(exists_)
    {
        LOG(debug) << "Bookmarks::number_key_pressed "
                  << " num "  << num 
                  << " modifiers " << Interactor::describeModifiers(modifiers) 
                  ; 

        setCurrent(num);
        apply();
    }
    else
    {
        if(Interactor::isShift(modifiers))
        {
            create(num);
        }
        else
        {
            LOG(info) << "Bookmarks::number_key_pressed no such bookmark  " << num << " (use shift modifier to create) " ; 
        }
    }
}


void Bookmarks::create(unsigned int num)
{
    setCurrent(num);
    LOG(info) << "Bookmarks::create : persisting state to slot " << m_current ; 
    collect();
    readdir();   // updates existance 
}



void Bookmarks::collect()
{
    if(m_current == UNSET ) return ; 

    if(m_verbose) LOG(info) << "Bookmarks::collect " << m_current ; 

    m_state->collect();
    m_state->setName(m_current);
    m_state->save();
}

void Bookmarks::apply()
{
    if(m_current == UNSET ) return ; 
    if(m_verbose) LOG(info) << "Bookmarks::apply " << m_current ; 

    m_state->setName(m_current);
    int rc = m_state->load();
    if(rc == 0)
    {
        m_state->apply();
    } 
    else
    {
        LOG(warning) << "Bookmarks::apply FAILED for m_current " << m_current ; 
        m_current = UNSET ; 
    } 
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


