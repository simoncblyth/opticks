#pragma once
/**
SGLFW.h : Trying to encapsulate OpenGL graphics with a light touch
====================================================================

Light touch encapsulation of OpenGL window and shader program, 
that means trying to hide boilerplate, but not making lots of 
decisions for user and getting complicated and inflexible like 
the old oglrap/Frame.hh oglrap/OpticksViz did. 
    
**/

#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#ifndef GLFW_TRUE
#define GLFW_TRUE true
#endif

#include "GL_CHECK.h"

#define GLEQ_IMPLEMENTATION
#include "gleq.h"

#include <glm/glm.hpp>
#include "SGLM.h"
#include "NPU.hh"

#include "SGLFW_Extras.h"
#include "SGLFW_Program.h"
#include "SGLFW_Render.h"

#ifdef WITH_CUDA_GL_INTEROP
#include "SGLFW_CUDA.h"
#endif

struct SGLFW : public SCMD 
{
    static constexpr const char* TITLE = "SGLFW" ; 

    SGLM& gm ; 
    int width ; 
    int height ; 

    const char* title ; 
    GLFWwindow* window ; 

    int count ; 
    int renderlooplimit ; 
    bool exitloop ; 

    bool dump ; 
    int  _width ;  // on retina 2x width 
    int  _height ;

    // getStartPos
    double _start_x ; 
    double _start_y ; 

    glm::vec2 start_ndc ;  // from key_pressed
    glm::vec2 move_ndc ;   // from cursor_moved
    glm::vec4 drag ; 

    SGLFW_Toggle toggle = {} ; 

    bool renderloop_proceed(); 
    void renderloop_exit(); 
    void renderloop_head(); 
    void renderloop_listen(); 
    void renderloop_tail(); 

    void handle_event(GLEQevent& event); 
    void key_pressed(unsigned key); 
    void key_released(unsigned key); 
    void cursor_moved(int ix, int iy); 
    void cursor_moved_action(); 
    int command(const char* cmd); 
    static std::string FormCommand(const char* token, float value); 

    void getWindowSize();
    std::string descWindowSize() const;

    void getStartPos(); 
    std::string descDrag() const;
    std::string descStartPos() const;  

    SGLFW(SGLM& gm, const char* title=nullptr ); 
    virtual ~SGLFW(); 
    static void Error_callback(int error, const char* description); 
    void init(); 

}; 

inline bool SGLFW::renderloop_proceed()
{
    return !glfwWindowShouldClose(window) && !exitloop ; 
}
inline void SGLFW::renderloop_exit()
{
    std::cout << "SGLFW::renderloop_exit" << std::endl; 
    glfwSetWindowShouldClose(window, true);
}
inline void SGLFW::renderloop_head()
{
    dump = count % 100000 == 0 ; 

    getWindowSize();
    glViewport(0, 0, _width, _height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if(dump) std::cout << "SGLFW::renderloop_head" << " gl.count " << count << std::endl ;
}

inline void SGLFW::renderloop_listen()
{
    GLEQevent event;
    while (gleqNextEvent(&event))
    {
        handle_event(event);
        gleqFreeEvent(&event);
    }
}

inline void SGLFW::renderloop_tail()
{
    glfwSwapBuffers(window);
    glfwPollEvents();
    exitloop = renderlooplimit > 0 && count++ > renderlooplimit ;
}



/**
SGLFW::handle_event
--------------------

See oglrap/Frame::handle_event

**/

inline void SGLFW::handle_event(GLEQevent& event)
{
    //std::cout << "SGLFW::handle_event " << event.type << std::endl; 
    switch(event.type)
    {
        case GLEQ_KEY_PRESSED:   key_pressed( event.keyboard.key)       ; break ; 
        case GLEQ_KEY_RELEASED:  key_released(event.keyboard.key)       ; break ;
        case GLEQ_CURSOR_MOVED:  cursor_moved(event.pos.x, event.pos.y) ; break ;
        default:                                                        ; break ; 
    }
}

inline void SGLFW::key_pressed(unsigned key)
{
    //std::cout << "SGLFW::key_pressed " << key << std::endl ;
    getStartPos(); 
    std::cout 
        << descStartPos() 
        << descWindowSize() 
        << std::endl
        ; 

    switch(key)
    {
        case GLFW_KEY_ESCAPE: command("--exit")      ; break ;  
        case GLFW_KEY_Z:      toggle.zoom = !toggle.zoom  ; break ; 
        case GLFW_KEY_N:      toggle.tmin = !toggle.tmin  ; break ; 
        case GLFW_KEY_F:      toggle.tmax = !toggle.tmax  ; break ; 
        case GLFW_KEY_R:      toggle.lrot = !toggle.lrot  ; break ; 
        case GLFW_KEY_C:      toggle.cuda = !toggle.cuda  ; break ; 
        case GLFW_KEY_A:      gm.command("--zoom 10")     ; break ; 
        case GLFW_KEY_D:      gm.command("--desc")        ; break ; 
    }

    std::cout << toggle.desc() << std::endl ; 

}
inline void SGLFW::key_released(unsigned key)
{
    //std::cout << "SGLFW::key_released " << key << std::endl ;
}


inline int SGLFW::command(const char* cmd)
{
    if(strcmp(cmd, "--exit") == 0) renderloop_exit(); 
    return 0 ;  
}

inline std::string SGLFW::FormCommand(const char* token, float value)  // static
{
    std::stringstream ss ; 
    ss << token << " " << value ; 
    std::string str = ss.str();
    return str ;
}




/**
SGLFW::getWindowSize
---------------------

eg on macOS with retina screen : SGLFW::descWindowSize wh[1024, 768] _wh[2048,1536]

**/
inline void SGLFW::getWindowSize()
{
    glfwGetWindowSize(window, &width, &height);
    glfwGetFramebufferSize(window, &_width, &_height);
}
inline std::string SGLFW::descWindowSize() const
{
    std::stringstream ss ; 
    ss << "SGLFW::descWindowSize"
       << " wh["
       << std::setw(4) << width
       << ","
       << std::setw(4) << height
       << "]"
       << " _wh["
       << std::setw(4) << _width 
       << ","
       << std::setw(4) << _height
       << "]"
       ;
    std::string str = ss.str();
    return str ;
}

/**
SGLFW::getStartPos
----------------------


::

    TL:cursor(0.,0.) ndc(-1.,1.)
    +-----------------+
    |                 |
    |                 |
    |        + CENTER: cursor(width/2,height/2) ndc(0.,0.)
    |                 |
    |                 |
    +-----------------+ BR:cursor(width, height) ndc(1.,-1.)


**/

inline void SGLFW::getStartPos()
{
    glfwGetCursorPos(window, &_start_x, &_start_y );

    start_ndc.x = 2.f*_start_x/width - 1.f ; 
    start_ndc.y = 1.f - 2.f*_start_y/height ; 
}

/**
SGLFW::cursor_moved
-----------------------

Follow old approach from Frame::cursor_moved_just_move

1. convert pixel positions into ndc (x,y) [-1:1, -1:1]


As cursor_moved gets called repeatedly during mouse
movements the drag.z drag.w tend to be small.

To combat this for local rotation control via quaternion
use the abolute start position (from key_pressed) 
and current position from cursor_moved. 

**/
inline void SGLFW::cursor_moved(int ix, int iy)
{
    move_ndc.x  = 2.f*float(ix)/width - 1.f ;   
    move_ndc.y  = 1.f - 2.f*float(iy)/height ; 

    float dx = move_ndc.x - drag.x ; 
    float dy = move_ndc.y - drag.y ;   // delta with the prior call to cursor_moved

    drag.x = move_ndc.x ; 
    drag.y = move_ndc.y ; 
    drag.z = dx ; 
    drag.w = dy ; 

    cursor_moved_action(); 
}

inline void SGLFW::cursor_moved_action()
{
    float dy = drag.w ; 
    if(toggle.zoom)
    {
        std::string cmd = FormCommand("--inc-zoom", dy*100 ); 
        gm.command(cmd.c_str()) ;
    } 
    else if(toggle.tmin)
    {
        std::string cmd = FormCommand("--inc-tmin", dy ); 
        gm.command(cmd.c_str()) ;
    }
    else if(toggle.tmax)
    {
        std::string cmd = FormCommand("--inc-tmax", dy ); 
        gm.command(cmd.c_str()) ;
    }
    else if(toggle.lrot)
    {
        gm.setLookRotation(start_ndc, move_ndc); 
        gm.update(); 
    }
}

inline std::string SGLFW::descDrag() const
{
    std::stringstream ss ; 
    ss
        << " (" 
        << std::setw(10) << std::fixed << std::setprecision(3) << drag.x 
        << ","
        << std::setw(10) << std::fixed << std::setprecision(3) << drag.y
        << ","
        << std::setw(10) << std::fixed << std::setprecision(3) << drag.z
        << ","
        << std::setw(10) << std::fixed << std::setprecision(3) << drag.w
        << ")" 
        ;
    std::string str = ss.str();
    return str ;
}


inline std::string SGLFW::descStartPos() const
{
    std::stringstream ss ; 
    ss << "SGLFW::descCursorPos"
       << "["
       << std::setw(7) << std::fixed << std::setprecision(2) << _start_x 
       << ","
       << std::setw(7) << std::fixed << std::setprecision(2) << _start_y
       << "]"
       << " ndc["
       << std::setw(7) << std::fixed << std::setprecision(2) << start_ndc.x
       << ","
       << std::setw(7) << std::fixed << std::setprecision(2) << start_ndc.y
       << "]"
       ;
    std::string str = ss.str();
    return str ;
}

inline SGLFW::SGLFW(SGLM& _gm, const char* title_ )
    :
    gm(_gm),
    width(gm.Width()),
    height(gm.Height()),
    title(title_ ? strdup(title_) : TITLE),
    window(nullptr),
    count(0),
    renderlooplimit(20000), 
    exitloop(false),
    dump(false),
    _width(0),
    _height(0),
    _start_x(0.),
    _start_y(0.),
    start_ndc(0.f,0.f),
    move_ndc(0.f,0.f),
    drag(0.f,0.f,0.f,0.f)
{
    init(); 
}

inline SGLFW::~SGLFW()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

inline void SGLFW::Error_callback(int error, const char* description) // static
{
    fprintf(stderr, "SGLFW::Error_callback: %s\n", description);
}

/**
SGLFW::init
-------------

1. OpenGL initialize
2. create window

Perhaps this needs to::

   glEnable(GL_DEPTH_TEST)


Example responses::

     Renderer: NVIDIA GeForce GT 750M OpenGL Engine
     OpenGL version supported 4.1 NVIDIA-10.33.0 387.10.10.10.40.105

     Renderer: TITAN RTX/PCIe/SSE2
     OpenGL version supported 4.1.0 NVIDIA 418.56

**/

inline void SGLFW::init()
{
    glfwSetErrorCallback(SGLFW::Error_callback);
    if (!glfwInit()) exit(EXIT_FAILURE);

    gleqInit();

#if defined __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);  // version specifies the minimum, not what will get on mac
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2); 
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#elif defined _MSC_VER
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4); 
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1); 
 
#elif __linux
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4); 
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);  // also used 6 here 
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // remove stuff deprecated in requested release
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint( GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);   // https://learnopengl.com/In-Practice/Debugging Debug output is core since OpenGL version 4.3,   
#endif


    GLFWmonitor* monitor = nullptr ;  // monitor to use for full screen mode, or NULL for windowed mode. 
    GLFWwindow* share = nullptr ;     // window whose context to share resources with, or NULL to not share resources

    window = glfwCreateWindow(width, height, title, monitor, share);
    if (!window)
    {   
        glfwTerminate();
        exit(EXIT_FAILURE);
    }   
    //glfwSetKeyCallback(window, SGLFW::key_callback);  // using gleq event for key callbacks not this manual approach 
    glfwMakeContextCurrent(window);

    gleqTrackWindow(window);  // replaces callbacks, see https://github.com/glfw/gleq
    
    glewExperimental = GL_TRUE;
    glewInit (); 

    

    GLenum err0 = glGetError() ; 
    GLenum err1 = glGetError() ; 
    bool err0_expected = err0 == GL_INVALID_ENUM ; // long-standing glew bug apparently 
    bool err1_expected = err1 == GL_NO_ERROR ; 
    if(!err0_expected) printf("//SGLFW::init UNEXPECTED err0 %d \n", err0 ); 
    if(!err1_expected) printf("//SGLFW::init UNEXPECTED err1 %d \n", err1 ); 
    //assert( err0_expected );  
    //assert( err1_expected );  

    const GLubyte* renderer = glGetString (GL_RENDERER);
    const GLubyte* version = glGetString (GL_VERSION);
    printf("//SGLFW::init renderer %s \n", renderer );
    printf("//SGLFW::init version %s \n", version );

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE); // otherwise gl_PointSize setting ignored, setting in geom not vert shader used when present 

    int interval = 1 ; // The minimum number of screen updates to wait for until the buffers are swapped by glfwSwapBuffers.
    glfwSwapInterval(interval);

}

