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


/*

Bookmarks:

* do not initiate actions

* record current state under a slot label 

* provide way to return to prior labelled state

* actions like navigating to a volume need to be implemented elsewhere 
  (Frame/Interactor/Scene)


*/


class Bookmarks {
public:
   static unsigned int N ; 
   static const char* filename ; 

public:
   Bookmarks();
   void setComposition(Composition* composition);
   void setScene(Scene* scene);
   void gui();

public:
   void number_key_pressed(unsigned int number, unsigned int modifiers=0);
   void number_key_released(unsigned int number);

public:
   void setCurrent(unsigned int num, bool create=false); 
   unsigned int getCurrent(); 
   void collect();  // collect config from associated objects into tree for current selected slot 
   void apply();    // apply config setting to associated objects using values from current selected slot of tree
   void add(unsigned int num); 

private: 
   void update();   // update existance array, done after loading bookmarks 
   void setup();    // setup existance array, done initially 

public:
   void dump(unsigned int num);
   bool exists(unsigned int num);
   bool exists_in_tree(unsigned int num);

   void roundtrip(const char* dir);  // save and load to debug fidelity 
   void load(const char* dir);       // populate m_tree from the file
   void save(const char* dir);       // write m_tree to the file

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

private:
   void dump(const char* name);
   void apply(const char* name);
   unsigned int collect(const char* name);
   unsigned int collectConfigurable(const char* name, Configurable* configurable);

private:
   int                           m_current ; 
   int                           m_current_gui ; 

   boost::property_tree::ptree   m_tree;
   Composition*                  m_composition ;
   Scene*                        m_scene ;
   Camera*                       m_camera ;
   View*                         m_view ;
   Trackball*                    m_trackball ;
   Clipper*                      m_clipper ;

   bool*                         m_exists ; 
   const char**                  m_names ;  

};






inline void Bookmarks::setScene(Scene* scene)
{
    m_scene = scene ; 
}


inline unsigned int Bookmarks::getCurrent()
{
    return m_current ; 
}



