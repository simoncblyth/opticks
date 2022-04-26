/**
UseGeometryShader
=================

Started from https://www.glfw.org/docs/latest/quick.html#quick_example

reference on geometry shaders https://open.gl/geometry


Review oglrap flying point rendering 
--------------------------------------

oglrap/gl/rec/geom.glsl::
  
    47 layout (lines) in;
    48 layout (points, max_vertices = 1) out;

    59     uint photon_id = gl_PrimitiveIDIn/MAXREC ;     // implies flat step-records are fed in 
    63     vec4 p0 = gl_in[0].gl_Position  ;
    64     vec4 p1 = gl_in[1].gl_Position  ; 



**/

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#ifndef GLFW_TRUE
#define GLFW_TRUE true
#endif

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include "SGLM.hh"
#include "NP.hh"

void check(const char* path, int line)
{
    GLenum err = glGetError() ;  
    bool ok = err == GL_NO_ERROR ;
    const char* s = NULL ; 
    switch(err)
    {   
        case GL_INVALID_ENUM: s = "GL_INVALID_ENUM" ; break ; 
        case GL_INVALID_VALUE: s = "GL_INVALID_VALUE" ; break ; 
        case GL_INVALID_OPERATION: s = "GL_INVALID_OPERATION" ; break ; 
        case GL_STACK_OVERFLOW : s = "GL_STACK_OVERFLOW" ; break ;   
        case GL_STACK_UNDERFLOW : s = "GL_STACK_UNDERFLOW" ; break ;   
        case GL_OUT_OF_MEMORY : s = "GL_OUT_OF_MEMORY" ; break ;   
        case GL_INVALID_FRAMEBUFFER_OPERATION : s = "GL_INVALID_FRAMEBUFFER_OPERATION" ; break ;
        case GL_CONTEXT_LOST : s = "GL_CONTEXT_LOST" ; break ;
    }   
    if(!ok) std::cout << "OpenGL ERROR " << path << " : " << line << " : " << std::hex << err << std::dec << " : " << s << std::endl ; 
    assert( ok );  
}

void print_shader_info_log(unsigned id)
{
    int max_length = 2048;
    int actual_length = 0;
    char log[2048];

    glGetShaderInfoLog(id, max_length, &actual_length, log);
    check(__FILE__, __LINE__ );  

    printf ("shader info log for GL index %u:\n%s\n", id, log);
    assert(0);
}
static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

static const char* SHADER_FOLD = getenv("SHADER_FOLD"); 
static const char* vertex_shader_text = NP::ReadString(SHADER_FOLD, "vert.glsl"); 
static const char* geometry_shader_text = NP::ReadString(SHADER_FOLD, "geom.glsl"); 
static const char* fragment_shader_text = NP::ReadString(SHADER_FOLD, "frag.glsl"); 

static const char* FOLD = getenv("FOLD"); 


int main()
{
    std::cout << " vertex_shader_text " << std::endl << vertex_shader_text << std::endl ; 
    std::cout << " geometry_shader_text " << std::endl << ( geometry_shader_text ? geometry_shader_text : "-" )  << std::endl ; 
    std::cout << " fragment_shader_text " << std::endl << fragment_shader_text << std::endl ; 

    NP* record = NP::Load(FOLD, "r.npy") ; 
    std::cout <<  " record " << record->sstr() << std::endl ; 
    GLsizei record_count = record->shape[0]*record->shape[1] ; 

    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) exit(EXIT_FAILURE);
    // version specifies the minimum
#if defined __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3); 
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2); 
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // this incantation gives
    //    Renderer: NVIDIA GeForce GT 750M OpenGL Engine
    //    OpenGL version supported 4.1 NVIDIA-10.33.0 387.10.10.10.40.105

#elif defined _MSC_VER
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);
 
#elif __linux
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // remove stuff deprecated in requested release
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint( GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);   // https://learnopengl.com/In-Practice/Debugging Debug output is core since OpenGL version 4.3,   
    
    // Frame::gl_init_window Renderer: TITAN RTX/PCIe/SSE2
    // Frame::gl_init_window OpenGL version supported 4.1.0 NVIDIA 418.56
#endif

    GLFWwindow* window;
    window = glfwCreateWindow(1024, 768, "UseGeometryShader", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwSetKeyCallback(window, key_callback);
    glfwMakeContextCurrent(window);
    
    glewExperimental = GL_TRUE;
    glewInit (); 
    assert( glGetError() == GL_INVALID_ENUM );  // long-standing glew bug apparently 
    assert( glGetError() == GL_NO_ERROR );  

    const GLubyte* renderer = glGetString (GL_RENDERER);
    const GLubyte* version = glGetString (GL_VERSION); 
    printf(" renderer %s \n", renderer ); 
    printf(" version %s \n", version ); 

    glfwSwapInterval(1);
    unsigned vao ;                                                              check(__FILE__, __LINE__); 
    glGenVertexArrays (1, &vao);                                                check(__FILE__, __LINE__);
    glBindVertexArray (vao);                                                    check(__FILE__, __LINE__);

    GLuint vertex_buffer ; 
    glGenBuffers(1, &vertex_buffer);                                            check(__FILE__, __LINE__);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);                               check(__FILE__, __LINE__);
    glBufferData(GL_ARRAY_BUFFER, record->arr_bytes(), record->bytes(), GL_STATIC_DRAW);  check(__FILE__, __LINE__);

    int params = -1; 
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);                    check(__FILE__, __LINE__);
    glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);                check(__FILE__, __LINE__);
    glCompileShader(vertex_shader);                                             check(__FILE__, __LINE__);
    glGetShaderiv (vertex_shader, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) print_shader_info_log(vertex_shader) ; 

    GLuint geometry_shader = 0 ;
    if( geometry_shader_text )
    {
        geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);                       check(__FILE__, __LINE__);
        glShaderSource(geometry_shader, 1, &geometry_shader_text, NULL);            check(__FILE__, __LINE__);
        glCompileShader(geometry_shader);                                           check(__FILE__, __LINE__);
        glGetShaderiv (geometry_shader, GL_COMPILE_STATUS, &params);
        if (GL_TRUE != params) print_shader_info_log(geometry_shader) ; 
    }

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);                check(__FILE__, __LINE__);
    glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);            check(__FILE__, __LINE__);
    glCompileShader(fragment_shader);                                           check(__FILE__, __LINE__);
    glGetShaderiv (fragment_shader, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) print_shader_info_log(fragment_shader) ; 

    GLuint program = glCreateProgram();        check(__FILE__, __LINE__);
    glAttachShader(program, vertex_shader);    check(__FILE__, __LINE__);
    if( geometry_shader > 0 ) glAttachShader(program, geometry_shader);  check(__FILE__, __LINE__);
    glAttachShader(program, fragment_shader);  check(__FILE__, __LINE__);
    glLinkProgram(program);                    check(__FILE__, __LINE__);

    GLint LightPosition_location       = glGetUniformLocation(program, "LightPosition");         check(__FILE__, __LINE__);
    GLint ModelViewProjection_location = glGetUniformLocation(program, "ModelViewProjection");   check(__FILE__, __LINE__);
    GLint Param_location               = glGetUniformLocation(program, "Param");                 check(__FILE__, __LINE__);


    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE); // otherwise gl_PointSize setting ignored 

    // grab locations from vert.glsl 
    GLint vpos_location = glGetAttribLocation(program, "vpos");  check(__FILE__, __LINE__);
    GLint vdir_location = glGetAttribLocation(program, "vdir");  check(__FILE__, __LINE__);
    GLint vcol_location = glGetAttribLocation(program, "vcol");  check(__FILE__, __LINE__);

    std::cout << " vpos_location " << vpos_location << std::endl ; 
    std::cout << " vdir_location " << vdir_location << std::endl ; 
    std::cout << " vcol_location " << vcol_location << std::endl ; 

    glUseProgram(program);

    GLsizei stride = sizeof(float)*4*4 ;  

    const void* vpos_offset = (void*)(sizeof(float)*0) ;   // pos
    const void* vdir_offset = (void*)(sizeof(float)*8) ;   // pol
    const void* vcol_offset = (void*)(sizeof(float)*8) ;   // pol 

    if( vpos_location > -1 )
    { 
        glEnableVertexAttribArray(vpos_location);                                              check(__FILE__, __LINE__);
        glVertexAttribPointer(vpos_location, 4, GL_FLOAT, GL_FALSE, stride, vpos_offset );     check(__FILE__, __LINE__);
    }

    if( vdir_location > -1 )
    {
        glEnableVertexAttribArray(vdir_location);                                             check(__FILE__, __LINE__);
        glVertexAttribPointer(vdir_location, 4, GL_FLOAT, GL_FALSE, stride, vdir_offset);     check(__FILE__, __LINE__);
    }

    if( vcol_location > -1 )
    {
        glEnableVertexAttribArray(vcol_location);                                             check(__FILE__, __LINE__);
        glVertexAttribPointer(vcol_location, 4, GL_FLOAT, GL_FALSE, stride, vcol_offset);     check(__FILE__, __LINE__);
    }

    bool exitloop(false);
    int renderlooplimit(2000);
    int count(0); 

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
 
    SGLM sglm ; 
    sglm.width = width ; 
    sglm.height = height ; 
    sglm.zoom = 1.f ;   

    sglm.eye_m.x = -2.f ; 
    sglm.eye_m.y = -2.f ; 
    sglm.eye_m.z = 0.f ; 

    sglm.setExtent(100.f); 
    sglm.update(); 
    sglm.dump(); 

    const glm::mat4& world2clip = sglm.world2clip ; 
    const GLfloat* mvp = (const GLfloat*) glm::value_ptr(world2clip) ;  

    //SGLM::GetMVP(width, height, verbose) ; 

    glm::vec4 LightPosition(0.f, 0.f, 0.f, 1.0);
    glm::vec4 Param(1.f, 0.f, 0.f, 0.0);

    while (!glfwWindowShouldClose(window) && !exitloop)
    {
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glUniformMatrix4fv(ModelViewProjection_location, 1, GL_FALSE, mvp );
        glUniform4fv(      LightPosition_location,       1, glm::value_ptr(LightPosition) );
        glUniform4fv(      Param_location,               1, glm::value_ptr(Param) );

        GLint first = 0 ; 
        glDrawArrays(GL_POINTS, first, record_count);
        glfwSwapBuffers(window);
        glfwPollEvents();
      
        exitloop = renderlooplimit > 0 && count > renderlooplimit ; 
        count++ ; 
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

