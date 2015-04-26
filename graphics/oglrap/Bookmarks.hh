#pragma once

#include <vector>
#include <string>
#include <map>

#include <boost/property_tree/ptree.hpp>

class Composition ;
class Camera ;
class View ;
class Scene ;


// unclear how best to arrange 
// maybe maintain maps of Camera and View 
// constructed based on the bookmark values
// and dispense those to the composition 

class Bookmarks {
public:

   Bookmarks()  
       :
       m_tree(),
       m_composition(NULL),
       m_scene(NULL),
       m_camera(NULL),
       m_view(NULL)
   {
   }

   void setComposition(Composition* composition);
   void setScene(Scene* scene);

   void apply(unsigned int number);
   void apply(const char* name);
   void load(const char* path);
   void save(const char* path);

private:
   boost::property_tree::ptree   m_tree;
   Composition*                  m_composition ;
   Scene*                        m_scene ;
   Camera*                       m_camera ;
   View*                         m_view ;


};



inline void Bookmarks::setScene(Scene* scene)
{
    m_scene = scene ; 
}



