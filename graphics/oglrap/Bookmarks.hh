#pragma once

#include <vector>
#include <string>
#include <map>

#include <boost/property_tree/ptree.hpp>

class Composition ;
class Camera ;
class View ;
class Trackball ; 
class Clipper ; 

class Scene ;
class Configurable ; 

// unclear how best to arrange 
// maybe maintain maps of Camera and View 
// constructed based on the bookmark values
// and dispense those to the composition 

class Bookmarks {
public:
   static const char* filename ; 

public:
   Bookmarks();
   void setComposition(Composition* composition);
   void setScene(Scene* scene);

public:
   void setCurrent(unsigned int num); 
   unsigned int getCurrent(); 

public:
   void apply(unsigned int num);
   void dump(unsigned int num);
   bool exists(unsigned int num);

   void number_key_pressed(unsigned int number, unsigned int container);
   void number_key_released(unsigned int number);

   void load(const char* dir);   // populate m_tree from the file
   void save(const char* dir);   // write m_tree to the file
   unsigned int collect(unsigned int num);  // collect changed values into m_tree under slot "num"


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
   unsigned int collect(const char* name);
   unsigned int collectConfigurable(const char* name, Configurable* configurable);

private:
   unsigned int                  m_current ; 
   boost::property_tree::ptree   m_tree;
   Composition*                  m_composition ;
   Scene*                        m_scene ;
   Camera*                       m_camera ;
   View*                         m_view ;
   Trackball*                    m_trackball ;
   Clipper*                      m_clipper ;


};



inline void Bookmarks::setScene(Scene* scene)
{
    m_scene = scene ; 
}

inline void Bookmarks::setCurrent(unsigned int num)
{
    m_current = num ; 
}
inline unsigned int Bookmarks::getCurrent()
{
    return m_current ; 
}



