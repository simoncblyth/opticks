#pragma once

#include <string>
#include <vector>
#include "plog/Severity.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GLEQ.hh"

#include <glm/fwd.hpp>  

#include "Touchable.hh"

class Opticks ; 

class Config ;
class Interactor ; 
class Composition ; 
class Scene ; 

struct Pix ; 


//  
// The window is created invisble by frame.init() 
// near the start of launch (this is necessary to create OpenGL context),
// then just before enter runloop the window is hinted visible and shown.
//  
// Unfortunately this trick does not work in fullscreen mode, 
// where a blank screen is presented until initialization completes.
//  
// TODO find solution/workaround
//  
// * when fullscreen is requested, start with an ordinary invisiblized window
//   then convert to fullscreen when context is ready 
//  
//   * :google:`glfw switch fullscreen`
//   * http://www.java-gaming.org/index.php?topic=34882.0
//   * https://github.com/glfw/glfw/issues/43  
//  
//     * window mode switching in pipeline for GLFW 3.2 ? which is a long way off
//     * Currently using 3.1.1 released March 2015
//  
//   * http://www.glfw.org/docs/3.0/context.html
//  
//     * :google:`GLFW context object sharing glfwCreateWindow`
//     * in GLFW3 the window is the context
//  
//  
// * splash screen with a progress message 
// * https://github.com/elliotwoods/ofxSplashScreen just presents an image, 
//    
// * trim some seconds off initialization


#include "OGLRAP_API_EXPORT.hh"
class OGLRAP_API Frame : public Touchable {
   public:
       static const plog::Severity LEVEL ; 
   public:
       Frame(Opticks* ok);
       virtual ~Frame();
       
       void configureI(const char* name, std::vector<int> values);
       void configureS(const char* name, std::vector<std::string> values);

       void setSize(std::string str);
       void setSize(unsigned int width, unsigned int height, unsigned int coord2pixel);
       glm::uvec4 getSize();

       void setTitle(const char* title);
       void setFullscreen(bool fullscreen);

       void toggleFullscreen_NOT_WORKING(bool fullscreen);

       void hintVisible(bool visible);
       void show();
       void init();

   private:
       void initHinting();
       void initContext();

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
       void viewport();
       void clear();
       GLFWwindow* getWindow();

   public:
       unsigned int getWidth(); 
       unsigned int getHeight();
       unsigned int getCoord2pixel();

   public:
       // Touchable 
       int touch(int ix, int iy);
   public:
       static float readDepth( int x, int y, int height );
       float readDepth( int x, int y);

       void snap();  

   private:
       void getCursorPos();
       void handle_event(GLEQevent& event);
       void dump_event(GLEQevent& event);
       void resize(unsigned int width, unsigned int height, unsigned int coord2pixel);

   private:
       void key_pressed(unsigned int key);
       void key_released(unsigned int key);

   private:
        enum { JUST_MOVE, CTRL_DRAG } ;
        void cursor_moved(GLEQevent& event);
        void cursor_moved_just_move(GLEQevent& event);
        void cursor_moved_ctrl_drag(GLEQevent& event);
   private:
       Opticks*      m_ok ;
       bool          m_fullscreen ;
       bool          m_is_fullscreen ;

       // TODO: eliminate most of these, get from composition
       unsigned int  m_width ; 
       unsigned int  m_height ; 
       unsigned int  m_width_prior ; 
       unsigned int  m_height_prior ;


       unsigned int  m_coord2pixel ; 
       const char*   m_title ;
       GLFWwindow*   m_window;

       Interactor*   m_interactor ; 
       Composition*  m_composition; 
       Scene*        m_scene ; 
       Pix*          m_pix ; 

       bool          m_cursor_inwindow ; 
       float         m_cursor_x ; 
       float         m_cursor_y ; 
       unsigned int  m_dumpevent ; 

   private:
       unsigned int  m_pixel_factor ; 
       // updated by getCursorPos
       int           m_pos_x ;
       int           m_pos_y ;

   private:
        unsigned m_cursor_moved_mode ; 


};





