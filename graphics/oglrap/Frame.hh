#ifndef FRAME_H
#define FRAME_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "gleq.h"

#include "Touchable.hh"
#include <string>
#include <vector>

class Config ;
class Interactor ; 
class Composition ; 
class Scene ; 


class Frame : public Touchable {
   public:
       Frame();
       virtual ~Frame();
       
       void configureI(const char* name, std::vector<int> values);
       void configureS(const char* name, std::vector<std::string> values);

       void setSize(unsigned int width, unsigned int height, unsigned int coord2pixel=2);
       void setTitle(const char* title);
       void setFullscreen(bool fullscreen);

       void hintVisible(bool visible);
       void show();
       void init();

   private:
       //void setPixelFactor(unsigned int factor);
       unsigned int getPixelFactor();
       void stipple();

   public:
       void setInteractor(Interactor* interactor);
       void setComposition(Composition* composition);
       void setScene(Scene* scene);

       void setDumpevent(int dumpevent);

       void exit();
    
       void listen();
       void render();
       GLFWwindow* getWindow();

   public:
       unsigned int getWidth(); 
       unsigned int getHeight();
       unsigned int getCoord2pixel();

   public:
       // Touchable 
       unsigned int touch(int ix, int iy);
   public:
       static float readDepth( int x, int y, int height );
       float readDepth( int x, int y);


   private:
       void getCursorPos();
       void handle_event(GLEQevent& event);
       void dump_event(GLEQevent& event);
       void resize(unsigned int width, unsigned int height);

   private:
       void key_pressed(unsigned int key);
       void key_released(unsigned int key);

   private:
       bool          m_fullscreen ;

       // TODO: eliminate most of these, get from composition
       unsigned int  m_width ; 
       unsigned int  m_height ; 
       unsigned int  m_coord2pixel ; 
       const char*   m_title ;
       GLFWwindow*   m_window;

       Interactor*   m_interactor ; 
       Composition*  m_composition; 
       Scene*        m_scene ; 

       bool          m_cursor_inwindow ; 
       float         m_cursor_x ; 
       float         m_cursor_y ; 
       unsigned int  m_dumpevent ; 

   private:
       unsigned int  m_pixel_factor ; 
       // updated by getCursorPos
       int           m_pos_x ;
       int           m_pos_y ;


};

inline Frame::Frame() : 
     m_fullscreen(false),
     m_title(NULL),
     m_window(NULL),
     m_interactor(NULL),
     m_cursor_inwindow(true),
     m_cursor_x(-1.f),
     m_cursor_y(-1.f),
     m_dumpevent(0),
     m_pixel_factor(1),
     m_pos_x(0),
     m_pos_y(0)
{
}


inline GLFWwindow* Frame::getWindow()
{ 
    return m_window ; 
}
inline unsigned int Frame::getWidth()
{  
    return m_width ; 
} 
inline unsigned int Frame::getHeight()
{ 
   return m_height ; 
} 
inline unsigned int Frame::getCoord2pixel()
{ 
   return m_coord2pixel ; 
} 



inline void Frame::setInteractor(Interactor* interactor)
{
    m_interactor = interactor ;
}
inline void Frame::setComposition(Composition* composition)
{
   m_composition = composition ; 
}
inline void Frame::setScene(Scene* scene)
{
   m_scene = scene ; 
}







#endif


