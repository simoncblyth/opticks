#pragma once

#include <vector>
#include <string>
#include <map>


class Composition ;
class View ; 
class Camera ; 


// unclear how best to arrange 
// maybe maintain maps of Camera and View 
// constructed based on the bookmark values
// and dispense those to the composition 

class Bookmarks {
public:

   Bookmarks() 
   {
   }

   void setComposition(Composition* composition);

   //void jump_to(const char* name);

   void load(const char* path);
   void save(const char* path);

private:
   Composition* m_composition ;

   std::map<std::string, View*>   m_views ; 
   std::map<std::string, Camera*> m_cameras ; 

};


inline void Bookmarks::setComposition(Composition* composition)
{
    m_composition = composition ; 
}


