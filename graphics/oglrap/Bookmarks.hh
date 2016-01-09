#pragma once

#include <map>
#include <vector>
#include <string>
#include <cstdio>

class NConfigurable ; 
class NState ; 

//
//  role of Bookmarks
//
//  manage swapping between states, 
//  states are persisted into .ini files within a directory 
//

class Bookmarks {
public:
   Bookmarks(NState* state);
   void gui();
private:
   void init();
public:
   // Interactor interface
   void number_key_pressed(unsigned int number, unsigned int modifiers=0);
   void number_key_released(unsigned int number);
public:
   bool exists(unsigned int num);
   unsigned int getCurrent(); 
   void setCurrent(unsigned int num); 


   void add(unsigned int num); 
   void collect();  // collect config from associated objects into tree for current selected slot 
   void apply();    // apply config setting to associated objects using values from current selected slot of tree


private:
   NState*                              m_state ; 
   int                                  m_current ; 
   int                                  m_current_gui ; 
   std::map<unsigned int, std::string>  m_bookmarks ;  

};



inline bool Bookmarks::exists(unsigned int num)
{
    return m_bookmarks.count(num) == 1 ; 
}

inline unsigned int Bookmarks::getCurrent()
{
    return m_current ; 
}

inline void Bookmarks::setCurrent(unsigned int num)
{
    if(exists(num))
    {
        m_current = num ; 
    }
    else
    {
        printf("Bookmarks::setCurrent no such bookmark %u\n", num);
    }
}




