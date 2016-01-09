// oglrap-
#include "Bookmarks.hh"
#include "Composition.hh"
#include "Camera.hh"
#include "View.hh"
#include "Trackball.hh"
#include "Clipper.hh"
#include "Interactor.hh"
#include "Scene.hh"


#include <cstring>
#include <string>
#include <set>
#include <exception>
#include <iostream>


#ifdef GUI_
#include <imgui.h>
#endif


// npy-
#include "jsonutil.hpp"
#include "NLog.hpp"


#ifdef OLD

#include <boost/property_tree/ini_parser.hpp>
#include <boost/foreach.hpp>

namespace pt = boost::property_tree;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

template<class Ptree>
inline const Ptree &empty_ptree()
{
   static Ptree pt;
   return pt;
}

#endif




unsigned int Bookmarks::N = 10 ; 
const char* Bookmarks::FILENAME = "bookmark.ini" ; 

Bookmarks::Bookmarks()  
       :
       m_current(0),
       m_current_gui(0),
       m_tree(),
       m_composition(NULL),
       m_scene(NULL),
       m_camera(NULL),
       m_view(NULL),
       m_trackball(NULL),
       m_clipper(NULL),
       m_exists(NULL),
       m_names(NULL)
{
    setup();
}

void Bookmarks::setup()
{
    m_exists = new bool[N] ;
    m_names  = new const char*[N] ;
    for(unsigned int i=0 ; i < N ; i++) 
    {
        char name[4];
        snprintf(name, 4, "%0.2d", i);
        m_names[i] = strdup(name);
        m_exists[i] = false ;
    }
}

void Bookmarks::update()
{
    for(unsigned int i=0 ; i < N ; i++) m_exists[i] = exists_in_tree(i);
}

bool Bookmarks::exists(unsigned int num)
{
    return num < N ? m_exists[num] : false ; 
}

bool Bookmarks::exists_in_tree(unsigned int num)
{
    std::string name = formName(num);
    return m_tree.count(name) > 0 ;
}

void Bookmarks::setCurrent(unsigned int num, bool create)
{
    assert(num < N);
    if(create) 
    {
        if(!m_exists[num]) m_exists[num] = true ;  
    } 

    if(m_exists[num])
    {
        m_current = num ; 
    }
    else
    {
        printf("Bookmarks::setCurrent no bookmark %u\n", num);
    }
}


void Bookmarks::load(const char* dir)
{
#ifdef OLD
    fs::path bookmarks(dir);
    bookmarks /= FILENAME ;

    try
    {
        pt::read_ini(bookmarks.string(), m_tree);
    }
    catch(const pt::ptree_error &e)
    {
        LOG(warning) << "Bookmarks::load ERROR " << e.what() ;
    }
#endif

    update(); // existance
    apply();  // bookmarks may have been externally changed
}

void Bookmarks::save(const char* dir)
{
#ifdef OLD
    fs::path bookmarks(dir);
    bookmarks /= FILENAME ;

    try
    {
        pt::write_ini(bookmarks.string(), m_tree);
    }
    catch(const pt::ptree_error &e)
    {
        LOG(warning) << "Bookmarks::save ERROR " << e.what() ;
    }

#endif
}


void Bookmarks::number_key_released(unsigned int num)
{
    LOG(info) << "Bookmarks::number_key_released " << num ; 
}

void Bookmarks::add(unsigned int num)
{
    bool create = true ; 
    setCurrent(num, create);
    collect();
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

void Bookmarks::roundtrip(const char* dir)
{
    LOG(info) << "Bookmarks::roundtrip " << dir ; 
    save(dir);
    load(dir);
}




void Bookmarks::setComposition(Composition* composition)
{
    m_composition = composition ; 
    m_camera = composition->getCamera();
    m_view   = composition->getView();
    m_trackball   = composition->getTrackball();
    m_clipper = composition->getClipper();
}

std::string Bookmarks::formName(unsigned int number)
{
    char name[32];
    snprintf(name, 32, "bookmark_%d", number );
    return name ;  
}

std::string Bookmarks::formKey(const char* name, const char* tag)
{
    char key[64];
    snprintf(key, 64, "%s.%s", name, tag );
    return key ;  
}



void Bookmarks::dump(unsigned int number)
{
    std::string name = formName(number);
    dump(name.c_str());
}



void Bookmarks::collect()
{
    if(m_current == 0 ) return ; 
    std::string name = formName(m_current);
    collect(name.c_str());
}
void Bookmarks::apply()
{
    if(m_current == 0 ) return ; 
    std::string name = formName(m_current);
    apply(name.c_str());
}


void Bookmarks::apply(const char* key, const char* val)
{
    if(View::accepts(key) && m_view)
    {
        m_view->configure(key, val);
    }
    else if(Camera::accepts(key) && m_camera)
    {
        m_camera->configure(key, val);
    } 
    else if(Trackball::accepts(key) && m_trackball)
    {
        m_trackball->configure(key, val);
    } 
    else if(Clipper::accepts(key) && m_clipper)
    {
        m_clipper->configure(key, val);
    }
    else if(Scene::accepts(key) && m_scene)
    {
        m_scene->configure(key, val);
    }
    else
    {
        LOG(debug) << "Bookmarks::apply ignoring  " 
                   << " key " << key  
                   << " val " << val  ; 

    }
}



void Bookmarks::apply(const char* name)
{
    BOOST_FOREACH( pt::ptree::value_type const& mk, m_tree.get_child("") ) 
    {   
        std::string mkk = mk.first;
        if(strcmp(name, mkk.c_str()) != 0) continue ; 

        LOG(info) << "Bookmarks::apply " << name ; 

        BOOST_FOREACH( pt::ptree::value_type const& it, mk.second.get_child("") ) 
        {   
            std::string itk = it.first;
            std::string itv = it.second.data();
            const char* key = itk.c_str();
            const char* val = itv.c_str();

            apply(key, val);
        }   
    }   
}


void Bookmarks::gui()
{
#ifdef GUI_

    const char* tmp = "/tmp" ;
    if(ImGui::Button("roundtrip")) roundtrip(tmp);
    ImGui::SameLine();
    if(ImGui::Button("tmpsave")) save(tmp);
    ImGui::SameLine();
    if(ImGui::Button("tmpload")) load(tmp);
    ImGui::SameLine();
    if(ImGui::Button("collect")) collect();
    ImGui::SameLine();
    if(ImGui::Button("apply")) apply();

    for(unsigned int i=0 ; i < N ; i++)
    {
        if(m_exists[i])
        {
             ImGui::RadioButton(m_names[i], &m_current_gui, i);
             ImGui::SameLine();
        }
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



unsigned int Bookmarks::collect(const char* name)
{
    // for bookmark "name" collect parameters of configurable objects into m_tree 
    // so doing this updates the bookmark : so long as succeed to save it 
    unsigned int changes(0);
    changes += collectConfigurable(name, m_view);
    changes += collectConfigurable(name, m_camera);
    changes += collectConfigurable(name, m_scene);
    changes += collectConfigurable(name, m_trackball);
    changes += collectConfigurable(name, m_clipper);
    return changes ; 
}


unsigned int Bookmarks::collectConfigurable(const char* name, NConfigurable* configurable)
{
    // Configurable is an abstract get/set/getTags/accepts/configure protocol 

    std::string empty ;
    std::vector<std::string> tags = configurable->getTags();

    unsigned int changes(0);

    for(unsigned int i=0 ; i < tags.size(); i++)
    {
        const char* tag = tags[i].c_str();
        std::string key = formKey(name, tag);
        std::string val = configurable->get(tag);    
        std::string prior = m_tree.get(key, empty);
        
        printf("Bookmarks::collectConfigurable %s : %s   prior : %s \n", key.c_str(), val.c_str(), prior.c_str() );

        if(prior.empty())
        {
            changes += 1 ; 
            m_tree.add(key, val);           
        }
        else if (strcmp(prior.c_str(), val.c_str())==0)
        {
            printf("unchanged key %s val %s \n", key.c_str(), val.c_str());
        }
        else
        {
            changes += 1 ; 
            m_tree.put(key, val);           
        }

    }
    return changes ; 
}


void Bookmarks::dump(const char* name)
{
    BOOST_FOREACH( pt::ptree::value_type const& mk, m_tree.get_child("") ) 
    {   
        std::string mkk = mk.first;
        if(strcmp(name, mkk.c_str()) != 0) continue ; 
        printf("%s\n", mkk.c_str());

        BOOST_FOREACH( pt::ptree::value_type const& it, mk.second.get_child("") ) 
        {   
            std::string itk = it.first;
            std::string itv = it.second.data();
            const char* key = itk.c_str();
            const char* val = itv.c_str();
            printf("    %10s : %s \n", key, val );  
        }
    }
}

void Bookmarks::update( const boost::property_tree::ptree& upt )
{
    BOOST_FOREACH( pt::ptree::value_type const& up, upt.get_child("") ) 
    {
        LOG(info)<<"Bookmarks::update " << up.first.data()  << " : " << up.second.data() ; 
        m_tree.put_child( up.first, up.second );
    }
}




