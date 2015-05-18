#pragma once

#include <vector>
#include <string>
#include <map>

#include <boost/property_tree/ptree.hpp>

class Composition ;
class Camera ;
class View ;
class Scene ;
class Configurable ; 

// unclear how best to arrange 
// maybe maintain maps of Camera and View 
// constructed based on the bookmark values
// and dispense those to the composition 

class Bookmarks {
public:
   static const char* filename ; 

   Bookmarks();
   void setComposition(Composition* composition);
   void setScene(Scene* scene);

   void apply(unsigned int number);
   void add(unsigned int number);

   void load(const char* dir);
   void save(const char* dir);

private:
   std::string formName(unsigned int number);
   std::string formKey(const char* name, const char* tag);

   void apply(const char* name);
   void add(const char* name);
   void addConfigurable(const char* name, Configurable* configurable);

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



