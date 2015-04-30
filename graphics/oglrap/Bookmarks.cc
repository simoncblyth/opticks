#include "Bookmarks.hh"
#include "Composition.hh"
#include "Camera.hh"
#include "View.hh"
#include "Scene.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include "string.h"

#include <boost/property_tree/ini_parser.hpp>
#include <boost/foreach.hpp>
#include <string>
#include <set>
#include <exception>
#include <iostream>

namespace pt = boost::property_tree;

void Bookmarks::load(const char* path)
{
   // transitionally : using the g4daeview.py bookmarks ini format 

   try
   {
        pt::read_ini(path, m_tree);
   }
   catch(const pt::ptree_error &e)
   {
        LOG(warning) << "Bookmarks::load ERROR " << e.what() ;
   }


}


void Bookmarks::setComposition(Composition* composition)
{
    m_composition = composition ; 
    m_camera = composition->getCamera();
    m_view   = composition->getView();
}


void Bookmarks::apply(unsigned int number)
{
    char name[32];
    snprintf(name, 32, "bookmark_%d", number );
    apply(name);
}


void Bookmarks::apply(const char* name)
{
    BOOST_FOREACH( boost::property_tree::ptree::value_type const& mk, m_tree.get_child("") ) 
    {   
        std::string mkk = mk.first;
        if(strcmp(name, mkk.c_str()) != 0) continue ; 

        LOG(info) << "Bookmarks::apply " << name ; 

        BOOST_FOREACH( boost::property_tree::ptree::value_type const& it, mk.second.get_child("") ) 
        {   
            std::string itk = it.first;
            std::string itv = it.second.data();
            const char* key = itk.c_str();
            const char* val = itv.c_str();

            if(View::accepts(key))
            {
                m_view->configure(key, val);
            }
            else if(Camera::accepts(key))
            {
                m_camera->configure(key, val);
            } 
            else if(Scene::accepts(key))
            {
                m_scene->configure(key, val);
            }
            else
            {
                LOG(warning) << "Bookmarks::apply ignoring  " 
                             << " name " << name 
                             << " key " << key  
                             << " val " << val  ; 

            }
        }   
    }   
}

void Bookmarks::save(const char* path)
{
    pt::write_ini(path, m_tree);
}


