#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <iostream>
#include <iomanip>

#include <boost/algorithm/string.hpp> 
#include <boost/lexical_cast.hpp>

#include "OGLRAP_BODY.hh"
#include "PLOG.hh"
#include "SPPM.hh"

//
//  C:\Program Files (x86)\Windows Kits\8.1\Include\shared\minwindef.h(130): warning C4005: 'APIENTRY': macro redefinition
// when PLOG is after glfw3

#include "Opticks.hh"


#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLEQ_IMPLEMENTATION
#include "GLEQ.hh"

#include "NGLM.hpp"

#include "Frame.hh"
#include "Interactor.hh"
#include "Composition.hh"
#include "Scene.hh"

#include "Pix.hh"


Frame::Frame(Opticks* ok) 
    :
     m_ok(ok), 
     m_fullscreen(false),
     m_is_fullscreen(false),
     m_width(0),
     m_height(0),
     m_width_prior(0),
     m_height_prior(0),
     m_coord2pixel(1),
     m_title(NULL),
     m_window(NULL),
     m_interactor(NULL),
     m_composition(NULL),
     m_scene(NULL),
     m_pix(new Pix),
     m_cursor_inwindow(true),
     m_cursor_x(-1.f),
     m_cursor_y(-1.f),
     m_dumpevent(0),
     m_pixel_factor(1),
     m_pos_x(0),
     m_pos_y(0),
     m_cursor_moved_mode(ok->hasOpt("ctrldrag") ? CTRL_DRAG : JUST_MOVE)
{
}


GLFWwindow* Frame::getWindow()
{ 
    return m_window ; 
}
unsigned int Frame::getWidth()
{  
    return m_width ; 
} 
unsigned int Frame::getHeight()
{ 
   return m_height ; 
} 
unsigned int Frame::getCoord2pixel()
{ 
   return m_coord2pixel ; 
} 


void Frame::setInteractor(Interactor* interactor)
{
    m_interactor = interactor ;
}
void Frame::setComposition(Composition* composition)
{
   m_composition = composition ; 
}
void Frame::setScene(Scene* scene)
{
   m_scene = scene ; 
}




void _update_fps_counter (GLFWwindow* window, const char* status) {
  if(!window)
  {
      printf("_update_fps_counter NULL window \n");
      return ; 
  }   
  static double previous_seconds = glfwGetTime ();
  static int frame_count;
  double current_seconds = glfwGetTime ();
  double elapsed_seconds = current_seconds - previous_seconds;
  if (elapsed_seconds > 0.25) {
    previous_seconds = current_seconds;
    double fps = (double)frame_count / elapsed_seconds;
    char tmp[128];
    sprintf (tmp, "%s fps: %.2f ", status, fps );
    glfwSetWindowTitle (window, tmp);
    frame_count = 0;
  }
  frame_count++;
}


static void error_callback(int /*error*/, const char* description)
{
    fputs(description, stderr);
}


Frame::~Frame()
{
    free((void*)m_title);
}

void Frame::setDumpevent(int dumpevent)
{
    m_dumpevent = dumpevent ; 
}

void Frame::configureI(const char* name, std::vector<int> values)
{
   if(values.empty()) return;
   int last = values.back(); 

   if(strcmp(name,"dumpevent")==0)
   {
       setDumpevent(last);
   }

}

void Frame::configureS(const char* name, std::vector<std::string> values)
{
   if(values.empty()) return;

   if(strcmp(name, "size") == 0)
   {
       std::string _whf = values.back();
       setSize(_whf);
   }
   else
   {
       printf("Frame::configureS param %s unknown \n", name);
   }
}

void Frame::setSize(std::string str)
{
    std::vector<std::string> whf;
    boost::split(whf, str, boost::is_any_of(","));
   
    if(whf.size() == 3 )
    {
        unsigned int width  = boost::lexical_cast<unsigned int>(whf[0]);  
        unsigned int height = boost::lexical_cast<unsigned int>(whf[1]);  
        unsigned int coord2pixel  = boost::lexical_cast<unsigned int>(whf[2]);  

        LOG(debug)<< "Frame::setSize" 
                 << " str " << str 
                 << " width " << width 
                 << " height " << height 
                 << " coord2pixel" << coord2pixel
                 ;
        setSize(width, height, coord2pixel);
    }
    else
    {
        LOG(warning)<< "Frame::setSize" 
                    << " str " << str 
                    << " str malformed : not a comma delimited triplet "
                    ;
          
    }
}


void Frame::setSize(unsigned int width, unsigned int height, unsigned int coord2pixel)
{
    LOG(debug) << "Frame::setSize "
              << " width " << width 
              << " height " << height 
              << " coord2pixel " << coord2pixel 
              ; 
    m_width = width ;
    m_height = height ;
    m_coord2pixel = coord2pixel ;

    m_pix->resize(width, height, coord2pixel); 
}

glm::uvec4 Frame::getSize()
{
   return glm::uvec4(m_width, m_height, m_coord2pixel, 0);
}



void Frame::setTitle(const char* title)
{
    m_title = strdup(title);
}

void Frame::setFullscreen(bool full)
{
    m_fullscreen = full ; 
}


void Frame::hintVisible(bool visible)
{
    glfwWindowHint(GLFW_VISIBLE, visible);
}

void Frame::show()
{
    glfwShowWindow(m_window);
}

void Frame::init()
{
    setSize(m_composition->getWidth(),m_composition->getHeight(),m_composition->getPixelFactor());

    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) ::exit(EXIT_FAILURE);

    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    LOG(debug) << "Frame::init " << m_width << "," << m_height << " " << m_title ;

    hintVisible(false);

    GLFWmonitor* monitor = m_fullscreen ? glfwGetPrimaryMonitor() : NULL ;  
    GLFWwindow* share = NULL ;  // window whose context to share resources with, or NULL to not share resources

    m_window = glfwCreateWindow(m_width, m_height, m_title, monitor, share );
    if (!m_window)
    {
        glfwTerminate();
        ::exit(EXIT_FAILURE);
    }


    glm::uvec4& position = m_composition->getFramePosition();
    glfwSetWindowPos(m_window, position.x, position.y );

    glfwMakeContextCurrent(m_window);

    initContext();  
}


void Frame::initContext()
{
    // hookup the callbacks and arranges outcomes into event queue 
    gleqTrackWindow(m_window);

    // start GLEW extension handler, segfaults if done before glfwCreateWindow
    glewExperimental = GL_TRUE;
    glewInit ();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);  // overwrite if distance to camera is less

    glClearColor(0.0,0.0,0.0,0.0);

    stipple();

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glfwSwapInterval(1);  // vsync hinting

    // get version info
    const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString (GL_VERSION); // version as a string
    LOG(debug) << "Frame::gl_init_window Renderer: " << renderer ;
    LOG(debug) << "Frame::gl_init_window OpenGL version supported " <<  version ;

    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    LOG(debug)<<"Frame::gl_init_window glfwGetFramebufferSize " << width << "," << height ;  
}


void Frame::toggleFullscreen_NOT_WORKING(bool fullscreen)
{
   // http://www.java-gaming.org/index.php?topic=34882.0
   //  http://www.glfw.org/docs/latest/monitor.html
   //   resolution of a video mode is specified in screen coordinates, not pixels.
   if( m_is_fullscreen == fullscreen )
   {
       LOG(info) << "Frame::toggleFullscreen already in that screen mode fullscreen? " << fullscreen  ;
       return ; 
   } 

   const GLFWvidmode* vm = glfwGetVideoMode(glfwGetPrimaryMonitor());

   LOG(info) << "Frame::toggleFullscreen VideoMode  " 
             << " width " << vm->width 
             << " height " << vm->height
             << " red " << vm->redBits
             << " green " << vm->greenBits
             << " blue " << vm->blueBits
             << " refresh " << vm->refreshRate
             << " fullscreen " << fullscreen 
             ;


    m_is_fullscreen = fullscreen ; 

    if(fullscreen)
    {
        m_width_prior = m_width ; 
        m_height_prior = m_height ; 
        m_width = vm->width ; 
        m_height = vm->height ; 
    }
    else
    {
        m_width = m_width_prior ; 
        m_height = m_height_prior ; 
    }

   
    // trying to keep the context alive while hopping between windows
 
    GLFWmonitor* monitor = fullscreen ? glfwGetPrimaryMonitor() : NULL ;  

    GLFWwindow* window = glfwCreateWindow(m_width, m_height, m_title, monitor, m_window );

    if (!window)
    {
        glfwTerminate();
        ::exit(EXIT_FAILURE);
    }

    glfwDestroyWindow(m_window);

    m_window = window ;  

    glfwMakeContextCurrent(m_window);

    initContext();  

}




void Frame::stipple()
{
    //https://www.opengl.org/archives/resources/faq/technical/transparency.htm#blen0025
    //https://www.opengl.org/sdk/docs/man2/xhtml/glPolygonStipple.xml
    //http://www.codeproject.com/Articles/23444/A-Simple-OpenGL-Stipple-Polygon-Example-EP-OpenGL

    GLubyte halftone[] = {
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55};

    glEnable(GL_POLYGON_STIPPLE);
    glPolygonStipple(halftone);
}

void Frame::listen()
{
    glfwPollEvents();

    GLEQevent event;
    while (gleqNextEvent(&event))
    {
        if(m_dumpevent) dump_event(event);
        handle_event(event);
        gleqFreeEvent(&event);
    }
}
 




void Frame::viewport()
{
     _update_fps_counter (m_window, m_interactor ? m_interactor->getStatus() : "" );

     // glViewport needs pixels (on retina)  window needs screen coordinates

     glViewport(0, 0, m_width*m_coord2pixel, m_height*m_coord2pixel);
}


void Frame::clear()
{
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}





void Frame::resize(unsigned int width, unsigned int height, unsigned int coord2pixel)
{
     if(width == 0 || height == 0) return ;  // ignore dud resizes

     setSize(width, height, coord2pixel);
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
     glViewport(0, 0, m_width*m_coord2pixel, m_height*m_coord2pixel);
}



int Frame::touch(int ix, int iy )
{
    float depth = readDepth(ix, iy );
    //LOG(info)<<"Frame::touch " << depth ;
    return m_scene ?  m_scene->touch(ix, iy, depth) : -1 ;
}


float Frame::readDepth( int x, int y )
{
   // window resize handling ?
    return readDepth( x, y, m_height*m_coord2pixel );
}


float Frame::readDepth( int x, int y_, int yheight )
{
    GLint y = yheight - y_ ;
    GLsizei width(1), height(1) ; 
    GLenum format = GL_DEPTH_COMPONENT ; 
    GLenum type = GL_FLOAT ;  
    float depth ; 
    glReadPixels(x, y, width, height, format, type, &depth ); 
    return depth ; 
}

void Frame::snap()
{
    m_pix->snap("/tmp/Frame.ppm"); 
}


void Frame::getCursorPos()
{
    double xpos ;  
    double ypos ;
    glfwGetCursorPos(m_window, &xpos, &ypos );
    m_pos_x = static_cast<int>(floor(xpos)*m_coord2pixel);
    m_pos_y = static_cast<int>(floor(ypos)*m_coord2pixel);
    //printf("Frame::getCursorPos    %d %d  (%d)    \n", m_pos_x, m_pos_y, m_coord2pixel );
}


void Frame::cursor_moved(GLEQevent& event)
{
    switch(m_cursor_moved_mode)
    {
        case JUST_MOVE: cursor_moved_just_move(event) ; break ;    
        case CTRL_DRAG: cursor_moved_ctrl_drag(event) ; break ;    
    }
}

void Frame::cursor_moved_just_move(GLEQevent& event)
{
     if(m_cursor_inwindow)
     {
          float cursor_dx = m_cursor_x > 0.f ? float(event.pos.x) - m_cursor_x : 0.f ; 
          float cursor_dy = m_cursor_y > 0.f ? float(event.pos.y) - m_cursor_y : 0.f ; 

          m_cursor_x = float(event.pos.x) ;
          m_cursor_y = float(event.pos.y) ;

          //printf("Cursor x,y %0.2f,%0.2f dx,dy  %0.2f,%0.2f \n", m_cursor_x, m_cursor_y, cursor_dx, cursor_dy );
          //
          // adjust to -1:1 -1:1 range with
          // 
          //       top right at (1,1)
          //       middle       (0,0)
          //       bottom left  (-1,-1)
          //

          float x = (2.f*m_cursor_x - m_width)/m_width ;
          float y = (m_height - 2.f*m_cursor_y)/m_height ;

          float dx = 2.f*cursor_dx/m_width  ;
          float dy = -2.f*cursor_dy/m_height ;

          // problem with this is how to end the drag, lifting finger and tapping 
          // screen comes over as sudden large drag, causing large trackball rotations
          //
          // so try a reset of the cursor position to being "undefined" when a jump is detected 
          //
          if(abs(dx) > 0.1 || abs(dy) > 0.1) 
          { 
              printf("jump? x,y (%0.5f,%0.5f)  dx,dy (%0.5f,%0.5f) \n", x, y, dx, dy );  
              m_cursor_x = -1.f ; 
              m_cursor_y = -1.f ; 
          }
          else
          {
              if(m_interactor)
              {
                  getCursorPos();
                  m_interactor->cursor_drag( x, y, dx, dy, m_pos_x, m_pos_y );
              }
          }
    }
}


void Frame::cursor_moved_ctrl_drag(GLEQevent& event)
{
        static bool flag_drag_done = false;
        // Using ctrl+click to control
        if ( 1 ) {
            static bool flag_drag_begin = false;
            if (event.type == GLEQ_BUTTON_PRESSED && event.button.button == 0 && event.button.mods == GLFW_MOD_CONTROL) {
                // printf("LT BUTTON PRESSED: mods: %d\n", event.button.mods);
                // save first time
                if (!flag_drag_begin) {
                    flag_drag_begin = true;
                    double _cursor_x, _cursor_y;
                    glfwGetCursorPos(m_window, &_cursor_x, &_cursor_y );
                    m_cursor_x = _cursor_x;
                    m_cursor_y = _cursor_y;
                }
            } else if (event.type == GLEQ_BUTTON_RELEASED && event.button.button == 0 && event.button.mods == GLFW_MOD_CONTROL) {
                if (flag_drag_begin) {
                // printf("LT BUTTON RELEASED: mods: %d\n", event.button.mods);
                flag_drag_done = true;
                flag_drag_begin = false;
                }
            }
        }

     if(m_cursor_inwindow && flag_drag_done)
     {
         double _cursor_x, _cursor_y;
         glfwGetCursorPos(m_window, &_cursor_x, &_cursor_y );
          float cursor_dx = m_cursor_x > 0.f ? _cursor_x - m_cursor_x : 0.f ; 
          float cursor_dy = m_cursor_y > 0.f ? _cursor_y - m_cursor_y : 0.f ; 
          m_cursor_x = _cursor_x ;
          m_cursor_y = _cursor_y ;
         //  float cursor_dx = m_cursor_x > 0.f ? float(event.pos.x) - m_cursor_x : 0.f ; 
         //  float cursor_dy = m_cursor_y > 0.f ? float(event.pos.y) - m_cursor_y : 0.f ; 

         // m_cursor_x = float(event.pos.x) ;
         // m_cursor_y = float(event.pos.y) ;

          //printf("Cursor x,y %0.2f,%0.2f dx,dy  %0.2f,%0.2f \n", m_cursor_x, m_cursor_y, cursor_dx, cursor_dy );
          //
          // adjust to -1:1 -1:1 range with
          // 
          //       top right at (1,1)
          //       middle       (0,0)
          //       bottom left  (-1,-1)
          //

          float x = (2.f*m_cursor_x - m_width)/m_width ;
          float y = (m_height - 2.f*m_cursor_y)/m_height ;

          float dx = 2.f*cursor_dx/m_width  ;
          float dy = -2.f*cursor_dy/m_height ;

          // problem with this is how to end the drag, lifting finger and tapping 
          // screen comes over as sudden large drag, causing large trackball rotations
          //
          // so try a reset of the cursor position to being "undefined" when a jump is detected 
          //
          if(abs(dx) > 0.1 || abs(dy) > 0.1) 
          { 
              printf("jump? x,y (%0.5f,%0.5f)  dx,dy (%0.5f,%0.5f) \n", x, y, dx, dy );  
              m_cursor_x = -1.f ; 
              m_cursor_y = -1.f ; 
          }
          else
          {
              if(m_interactor)
              {
                  getCursorPos();
                  // printf("jump? x,y (%0.5f,%0.5f)  dx,dy (%0.5f,%0.5f) \n", x, y, dx, dy );  
                  m_interactor->cursor_drag( x, y, dx, dy, m_pos_x, m_pos_y );
                  flag_drag_done = false; // after drag done, reset it
              }
          }
     } else if (m_cursor_inwindow) {
         // printf("LT touch pos (%d,%d)\n", m_pos_x, m_pos_y);
         getCursorPos();
         // touch(m_pos_x, m_pos_y);
         m_interactor->touch(m_pos_x, m_pos_y);
     }
}



void Frame::handle_event(GLEQevent& event)
{
    // some events like key presses scrub the position 
    //m_pos_x = floor(event.pos.x);
    //m_pos_y = floor(event.pos.y);
    //printf("Frame::handle_event    %d %d    \n", m_pos_x, m_pos_y );

    switch (event.type)
    {
        case GLEQ_FRAMEBUFFER_RESIZED:
            // printf("Frame::handle_event framebuffer resized to (%i %i)\n", event.size.width, event.size.height);
            resize(event.size.width, event.size.height, m_coord2pixel);
            break;
        case GLEQ_WINDOW_MOVED:
        case GLEQ_WINDOW_RESIZED:
            // printf("Frame::handle_event window resized to (%i %i)\n", event.size.width, event.size.height);
            resize(event.size.width, event.size.height, m_coord2pixel);
            break;
        case GLEQ_WINDOW_CLOSED:
        case GLEQ_WINDOW_REFRESH:
        case GLEQ_WINDOW_FOCUSED:
        case GLEQ_WINDOW_DEFOCUSED:
        case GLEQ_WINDOW_ICONIFIED:
        case GLEQ_WINDOW_RESTORED:
        case GLEQ_BUTTON_PRESSED:
        case GLEQ_BUTTON_RELEASED:
        case GLEQ_CURSOR_MOVED:
             cursor_moved(event);
             break;
        case GLEQ_SCROLLED:
             // FIXME
             // printf("Scrolled (%0.2f %0.2f)\n", event.pos.x, event.pos.y);
             if (1) {
                 float cursor_dx = float(event.pos.x); 
                 float cursor_dy = float(event.pos.y); 

                 float dx = 40.f*cursor_dx/m_width  ;
                 float dy = -40.f*cursor_dy/m_height;
                 float x = 1.;
                 float y = 1.;
                 if(m_cursor_inwindow)
                 {
                      if(m_interactor)
                      {
                          getCursorPos();
                          m_interactor->cursor_drag( x, y, dx, dy, m_pos_x, m_pos_y );
                      }
                 }
             }
             break;
        case GLEQ_CURSOR_ENTERED:
             m_cursor_inwindow = true ;
             LOG(debug)<< "Cursor entered window";
             break;
        case GLEQ_CURSOR_LEFT:
             m_cursor_inwindow = false ;
             LOG(debug) << "Cursor left window";
             break;
        // case GLEQ_SCROLLED:
        case GLEQ_KEY_PRESSED:
             key_pressed(event.key.key);
             break;

        case GLEQ_KEY_REPEATED:
        case GLEQ_KEY_RELEASED:
             key_released(event.key.key);
             break;

        case GLEQ_CHARACTER_INPUT:
        case GLEQ_FILE_DROPPED:
        case GLEQ_NONE:
            break;
    }
} 

void Frame::key_pressed(unsigned int key)
{
    if( key == GLFW_KEY_ESCAPE)
    {
        LOG(info)<<"Frame::key_pressed escape";
        glfwSetWindowShouldClose (m_window, 1);
    }
    else
    {
        LOG(debug)<<"Frame::key_pressed " <<  key ;
        getCursorPos();
        m_interactor->key_pressed(key);
    }

}  

void Frame::key_released(unsigned int key)
{
    LOG(debug)<<"Frame::key_released " <<  key ;
    getCursorPos();
    m_interactor->key_released(key);
}  
 


 
void Frame::dump_event(GLEQevent& event)
{
    switch (event.type)
    {
        case GLEQ_WINDOW_MOVED:
            printf("Window moved to (%.0f %.0f)\n", event.pos.x, event.pos.y);
            break;
        case GLEQ_WINDOW_RESIZED:
            printf("Window resized to (%i %i)\n", event.size.width, event.size.height);
            break;
        case GLEQ_WINDOW_CLOSED:
            printf("Window close request\n");
            break;
        case GLEQ_WINDOW_REFRESH:
            printf("Window refresh request\n");
            break;
        case GLEQ_WINDOW_FOCUSED:
            printf("Window focused\n");
            break;
        case GLEQ_WINDOW_DEFOCUSED:
            printf("Window defocused\n");
            break;
        case GLEQ_WINDOW_ICONIFIED:
            printf("Window iconified\n");
            break;
        case GLEQ_WINDOW_RESTORED:
            printf("Window restored\n");
            break;
        case GLEQ_FRAMEBUFFER_RESIZED:
            printf("Framebuffer resized to (%i %i)\n", event.size.width, event.size.height);
            break;
        case GLEQ_BUTTON_PRESSED:
            printf("Button %i pressed\n", event.button.button);
            break;
        case GLEQ_BUTTON_RELEASED:
            printf("Button %i released\n", event.button.button);
            break;
        case GLEQ_CURSOR_MOVED:
            printf("Cursor moved to (%0.2f %0.2f)\n", event.pos.x, event.pos.y);
            break;
        case GLEQ_CURSOR_ENTERED:
            LOG(debug)<<"Cursor entered window\n";
            break;
        case GLEQ_CURSOR_LEFT:
            LOG(debug)<<"Cursor left window\n";
            break;
        case GLEQ_SCROLLED:
            printf("Scrolled (%0.2f %0.2f)\n", event.pos.x, event.pos.y);
            break;
        case GLEQ_KEY_PRESSED:
            printf("Key 0x%02x pressed\n", event.key.key);
            break;
        case GLEQ_KEY_REPEATED:
            printf("Key 0x%02x repeated\n", event.key.key);
            break;
        case GLEQ_KEY_RELEASED:
            printf("Key 0x%02x released\n", event.key.key);
            break;
        case GLEQ_CHARACTER_INPUT:
            printf("Character 0x%08x input\n", event.character.codepoint);
            break;
        case GLEQ_FILE_DROPPED:
            printf("%i files dropped\n", event.file.count);
            for (int i = 0;  i < event.file.count;  i++)
                printf("\t%s\n", event.file.paths[i]);
            break;
        case GLEQ_NONE:
            break;
    }
}



void Frame::exit()
{
    glfwDestroyWindow(m_window);
    glfwTerminate();
}


