#include "Bookmarks.hh"
#include "Camera.hh"
#include "View.hh"

#include <boost/property_tree/ptree.hpp>
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

    pt::ptree tree;
    pt::read_ini(path, tree);

    BOOST_FOREACH( boost::property_tree::ptree::value_type const& mk, tree.get_child("") ) 
    {   
        std::string mkk = mk.first;
        std::cout << mkk << std::endl ;

        View* view = new View();
        Camera* camera = new Camera();

        BOOST_FOREACH( boost::property_tree::ptree::value_type const& it, mk.second.get_child("") ) 
        {   
            std::string itk = it.first;
            std::string itv = it.second.data();

            if(View::accepts(itk.c_str()))
            {
                view->configure(itk.c_str(), itv.c_str());
            }
            else if(Camera::accepts(itk.c_str()))
            {
                camera->configure(itk.c_str(), itv.c_str());
            } 
            std::cout << "   " << itk << " : " << itv << std::endl ;
        }   

        m_views[mkk] = view ; 
        m_cameras[mkk] = camera ; 

        view->Summary(mkk.c_str());
        camera->Summary(mkk.c_str());
    }   
}

void Bookmarks::save(const char* path)
{
    pt::ptree tree;
    pt::write_ini(path, tree);
}


