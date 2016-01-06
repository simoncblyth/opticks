#pragma once

#include <vector>
#include <string>
#include <map>


#define OLD 1

#ifdef OLD
#include <boost/property_tree/ptree.hpp>
#endif

class Composition ;
class Camera ;
class View ;
class Trackball ; 
class Clipper ; 

class Scene ;
class Configurable ; 

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
   static const char* FILENAME ; 

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
#ifdef OLD
    void update( const boost::property_tree::ptree& upt );
#else
    void update( const std::map<std::string,std::string>& upt );
#endif


private:
   std::string formName(unsigned int number);
   std::string formKey(const char* name, const char* tag);

private:
   void dump(const char* name);
   void apply(const char* name);
   void apply(const char* key, const char* val);
   unsigned int collect(const char* name);
   unsigned int collectConfigurable(const char* name, Configurable* configurable);

private:
   int                           m_current ; 
   int                           m_current_gui ; 

#ifdef OLD
   boost::property_tree::ptree   m_tree;
#else
   std::map<std::string, std::string> m_tree ;  
#endif



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



