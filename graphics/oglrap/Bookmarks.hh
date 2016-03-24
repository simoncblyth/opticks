#pragma once

#include <map>
#include <vector>
#include <string>
#include <cstdio>

class NConfigurable ; 
class NState ; 

class InterpolatedView ; 

//
//  *Bookmarks*
//
//       Manages swapping between states persisted into .ini files within a single *Bookmarks* directory
//       details handled by npy-/NState
//
//       Instantiating *Bookmarks* reads all the .ini state files within the directory into NState 
//       instances held in m_bookmarks std::map<unsigned int, NState*>
//
//       An InterpolatedView moving between Bookmarks can be created, to result in visible 
//       changes the viewpoint needs to differ between the states. Currently camera only differences
//       are not interpolated.
//
//       NB trackballed changes need to be collapsed into the view... 
// 
//
//

class Bookmarks {
public:
   enum { UNSET = -1, N=10 };
public:
   Bookmarks(const char* dir);
   void setState(NState* state);
   void setVerbose(bool verbose=true);
   void setInterpolatedViewPeriod(unsigned int ivperiod); 
   void create(unsigned int num);
   void gui();
   void refreshInterpolatedView();
   InterpolatedView* getInterpolatedView();
   std::string description(const char* msg="Bk");
   void Summary(const char* msg="Bookmarks::Summary");
private:
   void init();
   void readdir();
   void readmark(unsigned int num);
   void updateTitle();
   InterpolatedView* makeInterpolatedView();
   int parseName(const std::string& basename);
public:
   // Interactor interface
   void number_key_pressed(unsigned int number, unsigned int modifiers=0);
   void number_key_released(unsigned int number);
   void updateCurrent();
public:
   bool exists(unsigned int num);
   unsigned int getCurrent(); 
   const char* getTitle(); 
   void setCurrent(unsigned int num); 

   void collect();  // update state and persist to current slot, writing eg 001.ini
   void apply();    // instruct m_state to apply config setting to associated objects 

private:
   const char*                          m_dir ; 
   char                                 m_title[N+1] ;
   NState*                              m_state ; 
   InterpolatedView*                    m_view ;  
   int                                  m_current ; 
   int                                  m_current_gui ; 
   std::map<unsigned int, NState*>      m_bookmarks ;  
   bool                                 m_verbose ; 
   int                                  m_ivperiod ; 

};



inline const char* Bookmarks::getTitle()
{
   return &m_title[0] ; 
}

inline void Bookmarks::setVerbose(bool verbose)
{
   m_verbose = verbose ; 
}
inline void Bookmarks::setInterpolatedViewPeriod(unsigned int ivperiod)
{
   m_ivperiod = ivperiod ; 
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




