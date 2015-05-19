#include "Bookmarks.hh"
#include "Composition.hh"
#include "Camera.hh"
#include "View.hh"
#include "Trackball.hh"
#include "Clipper.hh"

#include "Scene.hh"

#include "string.h"

#ifdef COMPLEX
#include <boost/bind.hpp>
#endif

#include <boost/property_tree/ini_parser.hpp>
#include <boost/foreach.hpp>
#include <string>
#include <set>
#include <exception>
#include <iostream>



#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



namespace pt = boost::property_tree;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;



template<class Ptree>
inline const Ptree &empty_ptree()
{
   static Ptree pt;
   return pt;
}





const char* Bookmarks::filename = "bookmarks.ini" ; 

Bookmarks::Bookmarks()  
       :
       m_current(0),
       m_tree(),
       m_composition(NULL),
       m_scene(NULL),
       m_camera(NULL),
       m_view(NULL),
       m_trackball(NULL),
       m_clipper(NULL)
{
}



void Bookmarks::load(const char* dir)
{
    fs::path bookmarks(dir);
    bookmarks /= filename ;

    try
    {
        pt::read_ini(bookmarks.string(), m_tree);
    }
    catch(const pt::ptree_error &e)
    {
        LOG(warning) << "Bookmarks::load ERROR " << e.what() ;
    }

}

void Bookmarks::save(const char* dir)
{
    fs::path bookmarks(dir);
    bookmarks /= filename ;

    try
    {
        pt::write_ini(bookmarks.string(), m_tree);
    }
    catch(const pt::ptree_error &e)
    {
        LOG(warning) << "Bookmarks::save ERROR " << e.what() ;
    }
}

void Bookmarks::number_key_released(unsigned int num)
{
    LOG(info) << "Bookmarks::number_key_released " << num ; 

}
void Bookmarks::number_key_pressed(unsigned int num, unsigned int container)
{
    LOG(info) << "Bookmarks::number_key_pressed "
              << " num "  << num 
              << " container "  << container
              ; 

    if(exists(num))
    {
        dump(num);
        apply(num);
    }
    else
    {
       printf("no such bookmark yet\n");
    }

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

void Bookmarks::apply(unsigned int number)
{
    std::string name = formName(number);
    apply(name.c_str());
}
unsigned int Bookmarks::collect(unsigned int number)
{
    std::string name = formName(number);
    return collect(name.c_str());
}
void Bookmarks::dump(unsigned int number)
{
    std::string name = formName(number);
    dump(name.c_str());
}
bool Bookmarks::exists(unsigned int num)
{
    std::string name = formName(num);
    return m_tree.count(name) > 0 ;
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

            if(View::accepts(key))
            {
                m_view->configure(key, val);
            }
            else if(Camera::accepts(key))
            {
                m_camera->configure(key, val);
            } 
            else if(Trackball::accepts(key))
            {
                m_trackball->configure(key, val);
            } 
            else if(Clipper::accepts(key))
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
                             << " name " << name 
                             << " key " << key  
                             << " val " << val  ; 

            }
        }   
    }   
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


unsigned int Bookmarks::collectConfigurable(const char* name, Configurable* configurable)
{
    std::string empty ;
    std::vector<std::string> tags = configurable->getTags();

    unsigned int changes(0);

    for(unsigned int i=0 ; i < tags.size(); i++)
    {
        const char* tag = tags[i].c_str();
        std::string key = formKey(name, tag);
        std::string val = configurable->get(tag);    
        std::string prior = m_tree.get(key, empty);
        
        //printf("Bookmarks::collectConfigurable %s : %s   prior : %s \n", key.c_str(), val.c_str(), prior.c_str() );

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






#ifdef COMPLEX
// general soln not needed for simple ini format Bookmarks tree structure, 
// but for future more complex structures...
//
// http://stackoverflow.com/questions/8154107/how-do-i-merge-update-a-boostproperty-treeptree 
//
// The only limitation is that it is possible to have several nodes with the same
// path. Every one of them would be used, but only the last one will be merged.
//
// SO : restrict usage to unique key trees
//

void Bookmarks::complex_update(const boost::property_tree::ptree& pt) 
{
    traverse(pt, boost::bind(&Bookmarks::merge, this, _1, _2, _3));
}

template<typename T>
void Bookmarks::_traverse(
       const boost::property_tree::ptree &parent, 
       const boost::property_tree::ptree::path_type &childPath, 
       const boost::property_tree::ptree &child, 
       T method
       )
{
    method(parent, childPath, child);
    for(pt::ptree::const_iterator it=child.begin() ; it!=child.end() ;++it ) 
    {
        pt::ptree::path_type curPath = childPath / pt::ptree::path_type(it->first);
        _traverse(parent, curPath, it->second, method);
    }
}

template<typename T>
void Bookmarks::traverse(const boost::property_tree::ptree &parent, T method)
{
    _traverse(parent, "", parent, method);
}

void Bookmarks::merge(const boost::property_tree::ptree& parent, const boost::property_tree::ptree::path_type &childPath, const boost::property_tree::ptree &child) 
{
    LOG(info)<<"Bookmarks::merge " << childpath << " : " << child.data() ; 
    m_tree.put(childPath, child.data());
}    

#endif


 


