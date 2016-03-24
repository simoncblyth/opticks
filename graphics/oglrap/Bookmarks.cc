// oglrap-
#include "Bookmarks.hh"
#include "Interactor.hh"
#include "InterpolatedView.hh"

#include <cstring>
#include <sstream>

#ifdef GUI_
#include <imgui.h>
#endif

#include "regexsearch.hh"
#include <boost/lexical_cast.hpp>

// npy-
#include "dirutil.hpp"
#include "NState.hpp"
#include "NLog.hpp"

typedef std::map<unsigned int, NState*>::const_iterator MUSI ; 


Bookmarks::Bookmarks(const char* dir)  
       :
       m_dir(NULL),
       m_state(NULL),
       m_current(UNSET),
       m_current_gui(UNSET),
       m_view(NULL),
       m_verbose(false)
{
    std::string _dir = os_path_expandvars(dir) ;
    m_dir = strdup(_dir.c_str());
    init();
}


void Bookmarks::init()
{
    readdir();
}

void Bookmarks::setState(NState* state)
{
    m_state = state ; 
}


std::string Bookmarks::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " << getTitle() ; 
    return ss.str();
}

void Bookmarks::Summary(const char* msg)
{
    LOG(info) << description(msg);
}


void Bookmarks::updateTitle()
{
    m_title[N] = '\0' ;
    for(unsigned int i=0 ; i < N ; i++) m_title[i] = exists(i) ? i + '0' : '_' ; 
}


int Bookmarks::parseName(const std::string& basename)
{
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
    return num ; 
}



void Bookmarks::readdir()
{
    m_bookmarks.clear();

    typedef std::vector<std::string> VS ;
    VS basenames ; 
    dirlist(basenames, m_dir, ".ini" );  // basenames do not include the .ini

    LOG(debug) << "Bookmarks::readdir " << m_dir ;

    for(VS::const_iterator it=basenames.begin() ; it != basenames.end() ; it++)
    {
        std::string basename = *it ; 
        int num = parseName(basename); 
        if(num == UNSET) continue ; 
        readmark(num);
    }
    updateTitle();
}


void Bookmarks::readmark(unsigned int num)
{
    if(num == UNSET) return; 
    m_bookmarks[num] = NState::load(m_dir, num ) ; 
}


void Bookmarks::number_key_released(unsigned int num)
{
    LOG(debug) << "Bookmarks::number_key_released " << num ; 
}

void Bookmarks::number_key_pressed(unsigned int num, unsigned int modifiers)
{
    LOG(debug) << "Bookmarks::number_key_pressed "
               << " num "  << num 
               << " modifiers " << Interactor::describeModifiers(modifiers) 
               ; 

    bool shift = Interactor::isShift(modifiers) ;
    bool exists_ = exists(num);
    if(exists_)
    {
        if(num == m_current && shift)
        { 
            // repeating pressing a num key when on that bookmark with shift down
            LOG(info) << "Bookmarks::number_key_pressed repeat current book mark with shift" ;
            m_state->save();
            // the save updates, prior to persisting
        }
        else
        { 
            // set m_state name from m_current, load and apply : ie set values of attached configurables
            setCurrent(num);
            apply();  
        }
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
    readdir();   // clears and reloads all bookmarks, updating existance/states from the persisted files 
}

void Bookmarks::updateCurrent()
{
    bool exists_ = exists(m_current);
    if(exists_)
    {
        LOG(info) << "Bookmarks::updateCurrent persisting state " ;
        collect();
        readmark(m_current);
    }
    else
    {
        LOG(info) << "Bookmarks::updateCurrent m_current doesnt exist " << m_current ;
    }
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


    for(MUSI it=m_bookmarks.begin() ; it!=m_bookmarks.end() ; it++)
    {
         unsigned int num = it->first ; 
         std::string name = NState::FormName(num) ; 
         ImGui::RadioButton(name.c_str(), &m_current_gui, num);
    }

    // not directly setting m_current as need to notice a change
    if(m_current_gui != m_current ) 
    {
        setCurrent(m_current_gui);
        ImGui::Text(" changed : %d ", m_current);
        apply();
    }


    InterpolatedView* iv = getInterpolatedView();
    iv->gui();

#endif
}


InterpolatedView* Bookmarks::makeInterpolatedView()
{
    LOG(info) << "Bookmarks::makeInterpolatedView" ; 
    InterpolatedView* iv = new InterpolatedView ; 
    for(MUSI it=m_bookmarks.begin() ; it!=m_bookmarks.end() ; it++)
    {
         unsigned int num = it->first ; 
         NState* state = it->second ; 

         View* v = new View ; 
         state->addConfigurable(v);
         state->apply();

         iv->addView(v);
    }
    return iv ; 
}

void Bookmarks::refreshInterpolatedView()
{
    delete m_view ; 
    m_view = NULL ; 
}

InterpolatedView* Bookmarks::getInterpolatedView()
{
    if(!m_view) m_view = makeInterpolatedView();
    return m_view ;             
}


