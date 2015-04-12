
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLEQ_IMPLEMENTATION
#include "gleq.h"

#include "Frame.hh"
#include "Interactor.hh"

#include <iostream>
#include <iomanip>
#include <boost/algorithm/string.hpp> 
#include <boost/lexical_cast.hpp>


void _update_fps_counter (GLFWwindow* window) {
  static double previous_seconds = glfwGetTime ();
  static int frame_count;
  double current_seconds = glfwGetTime ();
  double elapsed_seconds = current_seconds - previous_seconds;
  if (elapsed_seconds > 0.25) {
    previous_seconds = current_seconds;
    double fps = (double)frame_count / elapsed_seconds;
    char tmp[128];
    sprintf (tmp, "opengl @ fps: %.2f", fps);
    glfwSetWindowTitle (window, tmp);
    frame_count = 0;
  }
  frame_count++;
}


static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}


Frame::Frame() : 
     m_title(NULL),
     m_window(NULL),
     m_interactor(NULL),
     m_cursor_inwindow(true),
     m_cursor_x(-1.f),
     m_cursor_y(-1.f),
     m_dumpevent(0)
{
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

       std::vector<std::string> whf;
       boost::split(whf, _whf, boost::is_any_of(","));
   
       if(whf.size() == 3 )
       {
           unsigned int width  = boost::lexical_cast<unsigned int>(whf[0]);  
           unsigned int height = boost::lexical_cast<unsigned int>(whf[1]);  
           unsigned int coord2pixel  = boost::lexical_cast<unsigned int>(whf[2]);  

           printf("Frame::configureS param %s : %s  \n", name, _whf.c_str());
           setSize(width, height, coord2pixel);
       }
       else
       {
           printf("Frame::configureS param %s malformed %s needs to be triplet eg 1024,768,2  \n", name, _whf.c_str());
       }
   }
   else
   {
       printf("Frame::configureS param %s unknown \n", name);
   }
}




void Frame::setSize(unsigned int width, unsigned int height, unsigned int coord2pixel)
{
    m_width = width ;
    m_height = height ;
    m_coord2pixel = coord2pixel ;
}
void Frame::setTitle(const char* title)
{
    m_title = strdup(title);
}
void Frame::setInteractor(Interactor* interactor)
{
    m_interactor = interactor ;
}





void Frame::gl_init_window()
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) ::exit(EXIT_FAILURE);

#ifdef  __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    m_window = glfwCreateWindow(m_width, m_height, m_title, NULL, NULL);
    if (!m_window)
    {
        glfwTerminate();
        ::exit(EXIT_FAILURE);
    }

    // hookup the callbacks and arranges outcomes into event queue 
    gleqTrackWindow(m_window);

    glfwMakeContextCurrent(m_window);

    // start GLEW extension handler, segfaults if done before glfwCreateWindow
    glewExperimental = GL_TRUE;
    glewInit ();

    glfwSwapInterval(1);  // vsync hinting

    // get version info
    const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString (GL_VERSION); // version as a string
    printf ("Frame::init_window Renderer: %s\n", renderer);
    printf ("Frame::init_window OpenGL version supported %s\n", version);

    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    printf("Frame::init_window FramebufferSize %d %d \n", width, height);    
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
 


void Frame::render()
{
     _update_fps_counter (m_window);

     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

     // glViewport needs pixels (on retina)  window needs screen coordinates
     glViewport(0, 0, m_width*m_coord2pixel, m_height*m_coord2pixel);

     //m_scene->draw(m_width, m_height);  
     // hmm only use of m_scene, maybe not stong enough cause for constituency
}


void Frame::resize(unsigned int width, unsigned int height)
{
     if(width == 0 || height == 0) return ;  // ignore dud resizes

     setSize(width, height);
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
     glViewport(0, 0, m_width*m_coord2pixel, m_height*m_coord2pixel);
}


void Frame::handle_event(GLEQevent& event)
{
    switch (event.type)
    {
        case GLEQ_FRAMEBUFFER_RESIZED:
            printf("Framebuffer resized to (%i %i)\n", event.size.width, event.size.height);
            resize(event.size.width, event.size.height);
            break;
        case GLEQ_WINDOW_MOVED:
        case GLEQ_WINDOW_RESIZED:
            printf("Window resized to (%i %i)\n", event.size.width, event.size.height);
            resize(event.size.width, event.size.height);
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
             if(m_cursor_inwindow)
             {
                  float cursor_dx = m_cursor_x > 0. ? event.pos.x - m_cursor_x : 0.f ; 
                  float cursor_dy = m_cursor_y > 0. ? event.pos.y - m_cursor_y : 0.f ; 

                  m_cursor_x = event.pos.x ;
                  m_cursor_y = event.pos.y ;

                  //printf("Cursor x,y %0.2f,%0.2f dx,dy  %0.2f,%0.2f \n", m_cursor_x, m_cursor_y, cursor_dx, cursor_dy );
                  //
                  // adjust to -1:1 -1:1 range with
                  // 
                  //       top right at (1,1)
                  //       middle       (0,0)
                  //       bottom left  (-1,-1)
                  //

                  float x = (2.*m_cursor_x - m_width)/m_width ;
                  float y = (m_height - 2.*m_cursor_y)/m_height ;
 
                  float dx = 2.*cursor_dx/m_width  ;
                  float dy = -2.*cursor_dy/m_height ;

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
                      if(m_interactor) m_interactor->cursor_drag( x, y, dx, dy );
                  }


             }
             break;
        case GLEQ_CURSOR_ENTERED:
             m_cursor_inwindow = true ;
             printf("Cursor entered window\n");
             break;
        case GLEQ_CURSOR_LEFT:
             m_cursor_inwindow = false ;
             printf("Cursor left window\n");
             break;
        case GLEQ_SCROLLED:
        case GLEQ_KEY_PRESSED:
             key_pressed(event.key.key );
             break;

        case GLEQ_KEY_REPEATED:
        case GLEQ_KEY_RELEASED:
             key_released(event.key.key );
             break;

        case GLEQ_CHARACTER_INPUT:
        case GLEQ_FILE_DROPPED:
        case GLEQ_NONE:
            break;
    }
} 

void Frame::key_pressed(unsigned int key)
{
    printf("Frame::key_pressed %u \n", key);

    if( key == GLFW_KEY_ESCAPE)
    {
        printf("Frame::key_pressed escape\n");
        glfwSetWindowShouldClose (m_window, 1);
    }
    else
    {
        m_interactor->key_pressed(key);
    }

}  

void Frame::key_released(unsigned int key)
{
    //printf("Frame::key_released %u \n", key);
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
            printf("Cursor entered window\n");
            break;
        case GLEQ_CURSOR_LEFT:
            printf("Cursor left window\n");
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


