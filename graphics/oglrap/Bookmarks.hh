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
   void dump(unsigned int number);

   void load(const char* dir);
   void save(const char* dir);


public:
    void update( const boost::property_tree::ptree& upt );


#ifdef COMPLEX
public:
    // general-ish ptree updating 
    // http://stackoverflow.com/questions/8154107/how-do-i-merge-update-a-boostproperty-treeptree 
    void complex_update(const boost::property_tree::ptree& pt);
protected:
    template<typename T>
    void  _traverse(const boost::property_tree::ptree& parent, const boost::property_tree::ptree::path_type& childPath, const boost::property_tree::ptree& child, T method);

    template<typename T>
    void traverse(const boost::property_tree::ptree &parent, T method);
    void merge(const boost::property_tree::ptree& parent, const boost::property_tree::ptree::path_type &childPath, const boost::property_tree::ptree &child);
#endif


private:
   std::string formName(unsigned int number);
   std::string formKey(const char* name, const char* tag);

   void dump(const char* name);
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



