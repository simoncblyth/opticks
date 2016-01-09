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
   enum { UNSET = -1, N=10 };
public:
   Bookmarks(NState* state);
   void setVerbose(bool verbose=true);
   void create(unsigned int num);
   void gui();
private:
   void init();
   void readdir();
   void updateTitle();
public:
   // Interactor interface
   void number_key_pressed(unsigned int number, unsigned int modifiers=0);
   void number_key_released(unsigned int number);
public:
   bool exists(unsigned int num);
   unsigned int getCurrent(); 
   const char* getTitle(); 
   void setCurrent(unsigned int num); 

   void collect();  // update state and persist to current slot, writing eg 001.ini
   void apply();    // instruct m_state to apply config setting to associated objects 

private:
   char                                 m_title[N+1] ;
   NState*                              m_state ; 
   int                                  m_current ; 
   int                                  m_current_gui ; 
   std::map<unsigned int, std::string>  m_bookmarks ;  
   bool                                 m_verbose ; 

};



inline const char* Bookmarks::getTitle()
{
   return &m_title[0] ; 
}

inline void Bookmarks::setVerbose(bool verbose)
{
   m_verbose = verbose ; 
}

inline bool Bookmarks::exists(unsigned int num)
{
    return m_bookmarks.count(num) == 1 ; 
}

inline void Bookmarks::setCurrent(unsigned int num)
{
    m_current = num ; 
}
inline unsigned int Bookmarks::getCurrent()
{
    return m_current ; 
}




