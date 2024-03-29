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

#ifdef WITH_CUDA_GL_INTEROP
#include "SCU.h"
#include "SCUDAOutputBuffer.h"
#include "SGLDisplay.h"
#endif

#define GLEQ_IMPLEMENTATION
#include "gleq.h"

#include <glm/glm.hpp>
#include "SGLM.h"
#include "NPU.hh"

#include "SGLFW_Extras.h"

struct SGLFW : public SCMD 
{
    static constexpr const char* TITLE = "SGLFW" ; 
    static constexpr const char* MVP_KEYS = "ModelViewProjection,MVP" ;  

    SGLM& gm ; 
    int width ; 
    int height ; 
#ifdef WITH_CUDA_GL_INTEROP
    SCUDAOutputBuffer<uchar4>* output_buffer ; 
    SGLDisplay* gl_display ; 
#endif
    const char* title ; 


    GLFWwindow* window ; 

    const char* vertex_shader_text ;
    const char* geometry_shader_text ; 
    const char* fragment_shader_text ;
    GLuint program ; 
    GLint  mvp_location ; 
    const float* mvp ; 
 
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
    void renderloop_update_state(); 
    void renderloop_tail(); 

    void listen(); 
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
    void init(); 

#ifdef WITH_CUDA_GL_INTEROP
    void fillOutputBuffer(); 
    void displayOutputBuffer();
#endif

    void createProgram(const char* _dir); 
    void createProgram(const char* vertex_shader_text, const char* geometry_shader_text, const char* fragment_shader_text ); 

    void enableAttrib( const char* name, const char* spec, bool dump=false ); 
    GLint getUniformLocation(const char* name) const ; 
    GLint findUniformLocation(const char* keys, char delim ) const ; 
    void locateMVP(const char* key, const float* mvp ); 
    void updateMVP();  // called from renderloop_head

    template<typename T>
    static std::string Desc(const T* tt, int num); 

    void UniformMatrix4fv( GLint loc, const float* vv ); 
    void Uniform4fv(       GLint loc, const float* vv ); 


    GLint getAttribLocation(const char* name) const ; 

    static void check(const char* path, int line); 
    static void print_shader_info_log(unsigned id); 
    static void error_callback(int error, const char* description); 
    //static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods); 
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

inline void SGLFW::renderloop_update_state()
{
    listen(); 
    updateMVP(); 
}

inline void SGLFW::listen()
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
#ifdef WITH_CUDA_GL_INTEROP
    output_buffer( nullptr ), 
    gl_display( nullptr ), 
#endif
    title(title_ ? strdup(title_) : TITLE),
    window(nullptr),
    vertex_shader_text(nullptr),
    geometry_shader_text(nullptr),
    fragment_shader_text(nullptr),
    program(0),
    mvp_location(-1),
    mvp(nullptr),
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
    glfwSetErrorCallback(SGLFW::error_callback);
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


#ifdef WITH_CUDA_GL_INTEROP
    output_buffer = new SCUDAOutputBuffer<uchar4>( SCUDAOutputBufferType::GL_INTEROP, width, height ) ; 
    std::cout << output_buffer->desc() ; 
    gl_display = new SGLDisplay ; 
    std::cout << gl_display->desc() ; 
#endif
}

#ifdef WITH_CUDA_GL_INTEROP

extern void SGLFW__fillOutputBuffer( dim3 numBlocks, dim3 threadsPerBlock, uchar4* output_buffer, int width, int height ); 
inline void SGLFW::fillOutputBuffer()
{
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    SCU::ConfigureLaunch2D(numBlocks, threadsPerBlock, output_buffer->width(), output_buffer->height() );   
    SGLFW__fillOutputBuffer(numBlocks, threadsPerBlock, 
         output_buffer->map(), 
         output_buffer->width(), 
         output_buffer->height() );           

    output_buffer->unmap();
    CUDA_SYNC_CHECK();
}

inline void SGLFW::displayOutputBuffer()
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //  
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );

    gl_display->display(
            output_buffer->width(),
            output_buffer->height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer->getPBO()
            );  
}
#endif

inline void SGLFW::createProgram(const char* _dir)
{
    const char* dir = U::Resolve(_dir); 

    vertex_shader_text = U::ReadString(dir, "vert.glsl"); 
    geometry_shader_text = U::ReadString(dir, "geom.glsl"); 
    fragment_shader_text = U::ReadString(dir, "frag.glsl"); 

    std::cout 
        << "SGLFW::createProgram" 
        << " _dir " << ( _dir ? _dir : "-" )
        << " dir "  << (  dir ?  dir : "-" )
        << " vertex_shader_text " << ( vertex_shader_text ? "YES" : "NO" ) 
        << " geometry_shader_text " << ( geometry_shader_text ? "YES" : "NO" ) 
        << " fragment_shader_text " << ( fragment_shader_text ? "YES" : "NO" ) 
        << std::endl 
        ;

    createProgram( vertex_shader_text, geometry_shader_text, fragment_shader_text ); 
}


/**
SGLFW::createProgram
---------------------

Compiles and links shader strings into a program referred from integer *program* 

On macOS with the below get "runtime error, unsupported version"::

    #version 460 core

On macOS with the below::

    #version 410 core

note that a trailing semicolon after the main curly brackets gives a syntax error, 
that did not see on Linux with "#version 460 core"

**/

inline void SGLFW::createProgram(const char* vertex_shader_text, const char* geometry_shader_text, const char* fragment_shader_text )
{
    std::cout << "[SGLFW::createProgram" << std::endl ; 
    //std::cout << " vertex_shader_text " << std::endl << vertex_shader_text << std::endl ;
    //std::cout << " geometry_shader_text " << std::endl << ( geometry_shader_text ? geometry_shader_text : "-" )  << std::endl ;
    //std::cout << " fragment_shader_text " << std::endl << fragment_shader_text << std::endl ;

    int params = -1;
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);                    SGLFW__check(__FILE__, __LINE__);
    glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);                SGLFW__check(__FILE__, __LINE__);
    glCompileShader(vertex_shader);                                             SGLFW__check(__FILE__, __LINE__);
    glGetShaderiv (vertex_shader, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) SGLFW::print_shader_info_log(vertex_shader) ;

    GLuint geometry_shader = 0 ;
    if( geometry_shader_text )
    {
        geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);                       SGLFW__check(__FILE__, __LINE__);
        glShaderSource(geometry_shader, 1, &geometry_shader_text, NULL);            SGLFW__check(__FILE__, __LINE__);
        glCompileShader(geometry_shader);                                           SGLFW__check(__FILE__, __LINE__);
        glGetShaderiv (geometry_shader, GL_COMPILE_STATUS, &params);
        if (GL_TRUE != params) SGLFW::print_shader_info_log(geometry_shader) ;
    }

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);                SGLFW__check(__FILE__, __LINE__);
    glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);            SGLFW__check(__FILE__, __LINE__);
    glCompileShader(fragment_shader);                                           SGLFW__check(__FILE__, __LINE__);
    glGetShaderiv (fragment_shader, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) SGLFW::print_shader_info_log(fragment_shader) ;

    program = glCreateProgram();               SGLFW__check(__FILE__, __LINE__);
    glAttachShader(program, vertex_shader);    SGLFW__check(__FILE__, __LINE__);
    if( geometry_shader > 0 )
    { 
        glAttachShader(program, geometry_shader); SGLFW__check(__FILE__, __LINE__);
    }
    glAttachShader(program, fragment_shader);  SGLFW__check(__FILE__, __LINE__);
    glLinkProgram(program);                    SGLFW__check(__FILE__, __LINE__);

    glUseProgram(program);

    std::cout << "]SGLFW::createProgram" << std::endl ; 
}


/**
SGLFW::enableAttrib
--------------------

Array attribute : connecting values from the array with attribute symbol in the shader program 

Example rpos spec "4,GL_FLOAT,GL_FALSE,64,0,false"


NB when handling multiple buffers note that glVertexAttribPointer
binds to the buffer object bound to GL_ARRAY_BUFFER when called. 
So that means have to repeatedly call this again after switching
buffers ? 

* https://stackoverflow.com/questions/14249634/opengl-vaos-and-multiple-buffers 
* https://antongerdelan.net/opengl/vertexbuffers.html

**/

inline void SGLFW::enableAttrib( const char* name, const char* spec, bool dump )
{
    SGLFW_Attrib att(name, spec); 

    att.index = getAttribLocation( name );     SGLFW__check(__FILE__, __LINE__);

    if(dump) std::cout << "SGLFW::enableArrayAttribute att.desc [" << att.desc() << "]" <<  std::endl ; 

    glEnableVertexAttribArray(att.index);      SGLFW__check(__FILE__, __LINE__);

    assert( att.integer_attribute == false ); 

    glVertexAttribPointer(att.index, att.size, att.type, att.normalized, att.stride, att.byte_offset_pointer );     SGLFW__check(__FILE__, __LINE__);
}


inline GLint SGLFW::getUniformLocation(const char* name) const 
{
    GLint loc = glGetUniformLocation(program, name);   SGLFW__check(__FILE__, __LINE__);
    return loc ; 
}
/**
SGLFW::findUniformLocation
---------------------------

Returns the location int for the first uniform key found in the 
shader program 

**/

inline GLint SGLFW::findUniformLocation(const char* keys, char delim ) const
{
    std::vector<std::string> kk ; 

    std::stringstream ss; 
    ss.str(keys)  ;
    std::string key;
    while (std::getline(ss, key, delim)) kk.push_back(key) ; 
 
    GLint loc = -1 ; 

    int num_key = kk.size(); 
    for(int i=0 ; i < num_key ; i++)
    {
        const char* k = kk[i].c_str(); 
        loc = getUniformLocation(k); 
        if(loc > -1) break ;  
    }
    return loc ; 
}


/**
SGLFW::locateMVP
------------------

Does not update GPU side, invoke SGLFW::locateMVP 
prior to the renderloop after shader program is 
setup and the GLM maths has been instanciated 
hence giving the pointer to the world2clip matrix
address. 

**/

inline void SGLFW::locateMVP(const char* key, const float* mvp_ )
{ 
    mvp_location = getUniformLocation(key); 
    assert( mvp_location > -1 ); 
    mvp = mvp_ ; 
}

/**
SGLFW::updateMVP
------------------

When mvp_location is > -1 this is called from 
the end of renderloop_head so any matrix updates
need to be done before then. 

**/

inline void SGLFW::updateMVP()
{
    assert( mvp_location > -1 ); 
    assert( mvp != nullptr ); 
    UniformMatrix4fv(mvp_location, mvp); 
}

template<typename T>
inline std::string SGLFW::Desc(const T* tt, int num) // static
{
    std::stringstream ss ; 
    for(int i=0 ; i < num ; i++) 
        ss  
            << ( i % 4 == 0 && num > 4 ? ".\n" : "" ) 
            << " " << std::fixed << std::setw(10) << std::setprecision(4) << tt[i] 
            << ( i == num-1 && num > 4 ? ".\n" : "" ) 
            ;   

    std::string s = ss.str(); 
    return s ; 
}

inline void SGLFW::UniformMatrix4fv( GLint loc, const float* vv )
{
    if(dump) std::cout 
        << "SGLFW::UniformMatrix4fv" 
        << " loc " << loc 
        << std::endl 
        << Desc(vv, 16) 
        << std::endl
        ;

    assert( loc > -1 ); 
    glUniformMatrix4fv(loc, 1, GL_FALSE, (const GLfloat*)vv );
}    

inline void SGLFW::Uniform4fv( GLint loc, const float* vv )
{
    if(dump) std::cout 
        << "SGLFW::Uniform4fv" 
        << " loc " << loc 
        << std::endl 
        << Desc(vv, 4) 
        << std::endl
        ;

    assert( loc > -1 ); 
    glUniform4fv(loc, 1, (const GLfloat*)vv );
}    

inline GLint SGLFW::getAttribLocation(const char* name) const 
{
    GLint loc = glGetAttribLocation(program, name);   SGLFW__check(__FILE__, __LINE__);
    return loc ; 
}

inline void SGLFW::print_shader_info_log(unsigned id)  // static
{
    int max_length = 2048;
    int actual_length = 0;
    char log[2048];

    glGetShaderInfoLog(id, max_length, &actual_length, log);
    SGLFW__check(__FILE__, __LINE__ );  

    printf("SGLFW::print_shader_info_log GL index %u:\n%s\n", id, log);
    assert(0);
}
inline void SGLFW::error_callback(int error, const char* description) // static
{
    fprintf(stderr, "SGLFW::error_callback: %s\n", description);
}


