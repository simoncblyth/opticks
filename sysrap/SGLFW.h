#pragma once
/**
SGLFW.h : Trying to encapsulate OpenGL graphics with a light touch
====================================================================

SGLFW_GLboolean
   string parse

SGLFW_bool
   string parse

SGLFW_GLenum
   string parse

SGLFW_Attribute
   parse attribute metadata strings such as "4,GL_FLOAT,GL_FALSE,64,0,false"

SGLFW
   light touch encapsulation of OpenGL window and shader program 
    
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




inline void SGLFW__check(const char* path, int line) // static
{
    GLenum err = glGetError() ;   
    bool ok = err == GL_NO_ERROR ;
    const char* s = NULL ; 
    switch(err)
    {   
        case GL_INVALID_ENUM:      s = "GL_INVALID_ENUM" ; break ; 
        case GL_INVALID_VALUE:     s = "GL_INVALID_VALUE" ; break ; 
        case GL_INVALID_OPERATION: s = "GL_INVALID_OPERATION" ; break ; 
        case GL_STACK_OVERFLOW:    s = "GL_STACK_OVERFLOW" ; break ;   
        case GL_STACK_UNDERFLOW:   s = "GL_STACK_UNDERFLOW" ; break ;   
        case GL_OUT_OF_MEMORY:     s = "GL_OUT_OF_MEMORY" ; break ;   
        case GL_CONTEXT_LOST:      s = "GL_CONTEXT_LOST" ; break ;
        case GL_INVALID_FRAMEBUFFER_OPERATION: s = "GL_INVALID_FRAMEBUFFER_OPERATION" ; break ;
    }   
    if(!ok) std::cout << "SGLFW__check OpenGL ERROR " << path << " : " << line << " : " << std::hex << err << std::dec << " : " << s << std::endl ; 
    assert( ok );  
}


struct SGLFW_GLboolean
{
    static constexpr const char* GL_FALSE_  = "GL_FALSE" ; 
    static constexpr const char* GL_TRUE_   = "GL_TRUE" ;
    static GLboolean Value(const char* name); 
    static const char* Name(GLboolean value); 
}; 
inline GLboolean SGLFW_GLboolean::Value(const char* name)
{
    GLboolean value = GL_FALSE ; 
    if( strcmp( name, GL_FALSE_) == 0 ) value = GL_FALSE ; 
    if( strcmp( name, GL_TRUE_)  == 0 ) value = GL_TRUE ; 
    return value ; 
}
inline const char* SGLFW_GLboolean::Name(GLboolean value)
{
    const char* s = nullptr ; 
    switch(value)
    {
       case GL_FALSE: s = GL_FALSE_ ; break ; 
       case GL_TRUE:  s = GL_TRUE_ ; break ; 
    }
    return s ; 
}


struct SGLFW_bool
{
    static constexpr const char* false_ = "false" ; 
    static constexpr const char* true_  = "true" ;
    static bool Value(const char* name); 
    static const char* Name(bool value); 
}; 
inline bool SGLFW_bool::Value(const char* name)
{
    bool value = false ; 
    if( strcmp( name, false_) == 0 ) value = false ; 
    if( strcmp( name, true_)  == 0 ) value = true ; 
    return value ; 
}
inline const char* SGLFW_bool::Name(bool value)
{
    return value ? true_ : false_ ; 
}


struct SGLFW_GLenum
{
    static constexpr const char* GL_BYTE_           = "GL_BYTE" ; 
    static constexpr const char* GL_UNSIGNED_BYTE_  = "GL_UNSIGNED_BYTE" ; 
    static constexpr const char* GL_SHORT_          = "GL_SHORT" ; 
    static constexpr const char* GL_UNSIGNED_SHORT_ = "GL_UNSIGNED_SHORT" ; 
    static constexpr const char* GL_INT_            = "GL_INT" ; 
    static constexpr const char* GL_UNSIGNED_INT_   = "GL_UNSIGNED_INT" ; 
    static constexpr const char* GL_HALF_FLOAT_     = "GL_HALF_FLOAT" ; 
    static constexpr const char* GL_FLOAT_          = "GL_FLOAT" ; 
    static constexpr const char* GL_DOUBLE_         = "GL_DOUBLE" ; 

    static const char* Name(GLenum type); 
    static GLenum      Type(const char* name); 
};

inline const char* SGLFW_GLenum::Name(GLenum type)
{
    const char* s = nullptr ; 
    switch(type)
    {
        case GL_BYTE:           s = GL_BYTE_           ; break ; 
        case GL_UNSIGNED_BYTE:  s = GL_UNSIGNED_BYTE_  ; break ; 
        case GL_SHORT:          s = GL_SHORT_          ; break ; 
        case GL_UNSIGNED_SHORT: s = GL_UNSIGNED_SHORT_ ; break ; 
        case GL_INT:            s = GL_INT_            ; break ; 
        case GL_UNSIGNED_INT:   s = GL_UNSIGNED_INT_   ; break ; 
        case GL_HALF_FLOAT:     s = GL_HALF_FLOAT_     ; break ;
        case GL_FLOAT:          s = GL_FLOAT_          ; break ;
        case GL_DOUBLE:         s = GL_DOUBLE_         ; break ;
        default:                s = nullptr            ; break ;
    }
    return s ; 
}

inline GLenum SGLFW_GLenum::Type(const char* name)
{
    GLenum type = 0 ; 
    if( strcmp( name, GL_BYTE_) == 0 )           type = GL_BYTE ; 
    if( strcmp( name, GL_UNSIGNED_BYTE_) == 0 )  type = GL_UNSIGNED_BYTE ; 
    if( strcmp( name, GL_SHORT_) == 0 )          type = GL_SHORT ; 
    if( strcmp( name, GL_UNSIGNED_SHORT_) == 0 ) type = GL_UNSIGNED_SHORT ; 
    if( strcmp( name, GL_INT_) == 0 )            type = GL_INT ; 
    if( strcmp( name, GL_UNSIGNED_INT_) == 0 )   type = GL_UNSIGNED_INT ; 
    if( strcmp( name, GL_HALF_FLOAT_) == 0 )     type = GL_HALF_FLOAT ; 
    if( strcmp( name, GL_FLOAT_) == 0 )          type = GL_FLOAT ; 
    if( strcmp( name, GL_DOUBLE_) == 0 )         type = GL_DOUBLE ; 
    return type ; 
}


/**
SGLFW_Attribute
-----------------

Parse a string of the below form into 6 fields::

    rpos:4,GL_FLOAT,GL_FALSE,64,0,false


**/
struct SGLFW_Attribute
{
    const char* name ; 
    const char* spec ; 
    std::vector<std::string> field ; 

    GLuint index ; 
    GLint size ;                   // field 0 : number of components of the attribute (aka item), must be one of : 1,2,3,4 
    GLenum type ;                  // field 1 : normally GL_FLOAT 
    GLboolean normalized ;         // field 2 : normalized means in range 0->1
    GLsizei stride ;               // field 3 : in bytes eg for 4,4 float photon/record struct stride is 4*4*4=64
    size_t   byte_offset ;         // field 4 : allows access to different parts of array of structs 
    bool     integer_attribute ;   // field 5       

    void*    byte_offset_pointer ; // derived from byte_offset 


    SGLFW_Attribute( const char* name, const char* spec ); 
    std::string desc() const ;  
};


inline SGLFW_Attribute::SGLFW_Attribute(const char* name_, const char* spec_)
    :
    name(strdup(name_)),
    spec(strdup(spec_)),
    index(0),
    size(0),
    type(0),
    normalized(false),
    stride(0),
    byte_offset(0),
    integer_attribute(false),
    byte_offset_pointer(nullptr)
{
    char delim = ',' ; 
    std::stringstream ss; 
    ss.str(spec)  ;
    std::string s;
    while (std::getline(ss, s, delim)) field.push_back(s) ; 
    assert( field.size() == 6 ); 

    size =  std::atoi(field[0].c_str()) ;           assert( size == 1 || size == 2 || size == 3 || size == 4 ) ; 
    type =  SGLFW_GLenum::Type(field[1].c_str()) ;  assert( type > 0 );    
    normalized = SGLFW_GLboolean::Value(field[2].c_str()) ; 
    stride = std::atoi( field[3].c_str() );          assert( stride > 0 ); 
    byte_offset = std::atoi( field[4].c_str() );     assert( byte_offset >= 0 ); 
    integer_attribute = SGLFW_bool::Value(field[5].c_str()) ; 

    byte_offset_pointer = (void*)byte_offset ; 
}

inline std::string SGLFW_Attribute::desc() const 
{
    std::stringstream ss ; 
    ss << "SGLFW_Attribute::desc" << std::endl 
       << std::setw(20) << "name"  << " : " << name << std::endl 
       << std::setw(20) << "spec"  << " : " << spec << std::endl 
       << std::setw(20) << "index" << " : " << index << std::endl 
       << std::setw(20) << "size"  << " : " << size << std::endl 
       << std::setw(20) << "type"  << " : " << SGLFW_GLenum::Name(type) << std::endl
       << std::setw(20) << "normalized" << " : " << SGLFW_GLboolean::Name(normalized) << std::endl
       << std::setw(20) << "stride" << " : " << stride << std::endl
       << std::setw(20) << "byte_offset" << " : " << byte_offset << std::endl
       << std::setw(20) << "integer_attribute" << " : " << SGLFW_bool::Name(integer_attribute) << std::endl
       << std::setw(20) << "byte_offset_pointer" << " : " << byte_offset_pointer << std::endl
       ;

    for(unsigned i=0 ; i < field.size() ; i++ ) ss << std::setw(20) << i << " : " << field[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}


/**
SGLFW_Buffer
-------------

Old Opticks oglrap handled multi-buffers using RBuf held by Renderer
See::

   Renderer::createVertexArray

**/


struct SGLFW_Buffer
{
    GLuint id ; 
    SGLFW_Buffer( int num_bytes, const void* data, GLenum target, GLenum usage  ); 
};

SGLFW_Buffer::SGLFW_Buffer( int num_bytes, const void* data , GLenum target, GLenum usage )
{
    glGenBuffers(1, &id );                         SGLFW__check(__FILE__, __LINE__);
    glBindBuffer(target, id);                      SGLFW__check(__FILE__, __LINE__);     
    glBufferData(target, num_bytes, data, usage ); SGLFW__check(__FILE__, __LINE__);
}



/**
SGLFW
------

Light touch encapsulation of OpenGL window and shader program 
(light touch means: trying to hide boilerplate, not making lots of decisions for user
and getting complicated like oglrap did)

**/

struct SGLFW
{
    static constexpr const char* TITLE = "SGLFW" ; 

    GLFWwindow* window ; 
    int width ; 
    int height ; 
    const char* title ; 

    GLuint program ; 

    SGLFW(int width, int height, const char* title=nullptr ); 
    virtual ~SGLFW(); 

    void init(); 
    void createProgram(const char* vertex_shader_text, const char* geometry_shader_text, const char* fragment_shader_text ); 

    void enableArrayAttribute( const char* name, const char* spec ); 
    GLint getUniformLocation(const char* name) const ; 
    GLint getAttribLocation(const char* name) const ; 


    static void check(const char* path, int line); 
    static void print_shader_info_log(unsigned id); 
    static void error_callback(int error, const char* description); 
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods); 
}; 


inline SGLFW::SGLFW(int width_, int height_, const char* title_ )
    :
    width(width_),
    height(height_),
    title(title_ ? strdup(title_) : TITLE),
    program(0)
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
    glfwSetKeyCallback(window, SGLFW::key_callback);
    glfwMakeContextCurrent(window);
    
    glewExperimental = GL_TRUE;
    glewInit (); 

    GLenum err0 = glGetError() ; 
    GLenum err1 = glGetError() ; 
    bool err0_expected = err0 == GL_INVALID_ENUM ; // long-standing glew bug apparently 
    bool err1_expected = err1 == GL_NO_ERROR ; 
    if(!err0_expected) printf("//SGLFW::init UNEXPECTED err0 %d \n", err0 ); 
    if(!err1_expected) printf("//SGLFW::init UNEXPECTED err1 %d \n", err1 ); 
    assert( err0_expected );  
    assert( err1_expected );  

    const GLubyte* renderer = glGetString (GL_RENDERER);
    const GLubyte* version = glGetString (GL_VERSION);
    printf("//SGLFW::init renderer %s \n", renderer );
    printf("//SGLFW::init version %s \n", version );

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE); // otherwise gl_PointSize setting ignored, setting in geom not vert shader used when present 

    int interval = 1 ; // The minimum number of screen updates to wait for until the buffers are swapped by glfwSwapBuffers.
    glfwSwapInterval(interval);
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
    if( geometry_shader > 0 ) glAttachShader(program, geometry_shader);  SGLFW__check(__FILE__, __LINE__);
    glAttachShader(program, fragment_shader);  SGLFW__check(__FILE__, __LINE__);
    glLinkProgram(program);                    SGLFW__check(__FILE__, __LINE__);

    glUseProgram(program);
    std::cout << "]SGLFW::createProgram" << std::endl ; 
}




/**
SGLFW::enableArrayAttribute
------------------------------

Array attribute : connecting values from the array with attribute symbol in the shader program 

Example rpos spec "4,GL_FLOAT,GL_FALSE,64,0,false"


NB when handling multiple buffers note that glVertexAttribPointer
binds to the buffer object bound to GL_ARRAY_BUFFER when called. 
So that means have to repeatedly call this again after switching
buffers ? 

* https://stackoverflow.com/questions/14249634/opengl-vaos-and-multiple-buffers 
* https://antongerdelan.net/opengl/vertexbuffers.html

**/

void SGLFW::enableArrayAttribute( const char* name, const char* spec )
{
    SGLFW_Attribute att(name, spec); 

    att.index = getAttribLocation( name );     SGLFW__check(__FILE__, __LINE__);

    std::cout << "SGLFW::enableArrayAttribute att.desc [" << att.desc() << "]" <<  std::endl ; 

    glEnableVertexAttribArray(att.index);      SGLFW__check(__FILE__, __LINE__);

    assert( att.integer_attribute == false ); 

    glVertexAttribPointer(att.index, att.size, att.type, att.normalized, att.stride, att.byte_offset_pointer );     SGLFW__check(__FILE__, __LINE__);
}


GLint SGLFW::getUniformLocation(const char* name) const 
{
    GLint loc = glGetUniformLocation(program, name);   SGLFW__check(__FILE__, __LINE__);
    return loc ; 
}

GLint SGLFW::getAttribLocation(const char* name) const 
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

/**

some ideas on key handling :  UseOpticksGLFW/UseOpticksGLFW.cc 

https://stackoverflow.com/questions/55573238/how-do-i-do-a-proper-input-class-in-glfw-for-a-game-engine

https://learnopengl.com/Getting-started/Camera

THIS NEED TO TALK TO SGLM::INSTANCE changing viewpoint 

**/
inline void SGLFW::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) // static
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {   
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }   
}



