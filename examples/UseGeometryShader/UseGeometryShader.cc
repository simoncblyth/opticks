/**
UseGeometryShader
=================

Started from https://www.glfw.org/docs/latest/quick.html#quick_example


reference on geometry shaders https://open.gl/geometry


**/

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#ifndef GLFW_TRUE
#define GLFW_TRUE true
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

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


static const char* vertex_shader_text = R"glsl(
#version 410 core

// axis passthrough to geometry shader

uniform mat4 ModelViewProjection ;
//uniform mat4 ModelView ;
uniform vec4 LightPosition ; 
uniform vec4 Param ;

layout(location = 0) in vec4 vpos ;
layout(location = 1) in vec4 vdir ;
layout(location = 2) in vec4 vcol ;

out vec3 position ; 
out vec3 direction ; 
out vec3 colour ; 

void main ()  
{
    colour = vec3(vcol) ;

    position = vpos.xyz ;

    direction = vdir.xyz ;

    gl_Position = vec4( vec3( LightPosition ) , 1.0);
}

)glsl";


static const char* geometry_shader_text = R"glsl(
#version 410 core

//uniform mat4 Projection ;
uniform mat4 ModelViewProjection ;
uniform vec4 Param ; 

in vec3 direction[];
in vec3 colour[];

layout (points) in; 
layout (line_strip, max_vertices = 2) out;

out vec3 fcolour ; 


void main ()  
{
    gl_Position = ModelViewProjection * gl_in[0].gl_Position ;
    fcolour = colour[0] ;
    EmitVertex();

    gl_Position = ModelViewProjection * ( gl_in[0].gl_Position + Param.x*vec4(direction[0], 0.) ) ; 
    fcolour = colour[0] ;
    EmitVertex();

    EndPrimitive();
} 

)glsl";


static const char* fragment_shader_text = R"glsl(
#version 410 core

in vec3 fcolour;
out vec4 frag_colour;

void main ()  
{
   frag_colour = vec4(fcolour, 1.0);
}

)glsl";



static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}
int main(void)
{
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
    window = glfwCreateWindow(640, 480, "UseGeometryShader", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwSetKeyCallback(window, key_callback);
    glfwMakeContextCurrent(window);
    
    //gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
    glewExperimental = GL_TRUE;
    glewInit (); 
    assert( glGetError() == GL_INVALID_ENUM );  // long-standing glew bug apparently 
    assert( glGetError() == GL_NO_ERROR );  

    const GLubyte* renderer = glGetString (GL_RENDERER);
    const GLubyte* version = glGetString (GL_VERSION); 
    printf(" renderer %s \n", renderer ); 
    printf(" version %s \n", version ); 

    glfwSwapInterval(1);

    unsigned vao ; 
    check(__FILE__, __LINE__);
    glGenVertexArrays (1, &vao);                                                check(__FILE__, __LINE__);
    glBindVertexArray (vao);                                                    check(__FILE__, __LINE__);


    const glm::vec4 vertices[] = {
        glm::vec4( 0.f   ,    0.f,    0.f, 0.f ), 
        glm::vec4( 1000.f,    0.f,    0.f, 0.f ), 
        glm::vec4( 1.f   ,    0.f,    0.f, 0.f ), 
        glm::vec4( 0.f   ,    0.f,    0.f, 0.f ), 
        glm::vec4( 0.f   ,    1.f,    0.f, 0.f ), 
        glm::vec4( 0.f   , 1000.f,    0.f, 0.f ), 
        glm::vec4( 0.f   ,    0.f,    0.f, 0.f ),
        glm::vec4( 0.f   ,    0.f, 1000.f, 0.f ),
        glm::vec4( 0.f   ,    0.f,    1.f, 0.f )
    }; 
    assert( sizeof(vertices)/sizeof(glm::vec4) == 9 ); 


    GLuint vertex_buffer ; 
    glGenBuffers(1, &vertex_buffer);                                            check(__FILE__, __LINE__);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);                               check(__FILE__, __LINE__);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);  check(__FILE__, __LINE__);


    int params = -1; 
    GLuint vertex_shader ;
    vertex_shader = glCreateShader(GL_VERTEX_SHADER);                           check(__FILE__, __LINE__);
    glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);                check(__FILE__, __LINE__);
    glCompileShader(vertex_shader);                                             check(__FILE__, __LINE__);
    glGetShaderiv (vertex_shader, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) print_shader_info_log(vertex_shader) ; 

    GLuint geometry_shader ; 
    geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);                         check(__FILE__, __LINE__);
    glShaderSource(geometry_shader, 1, &geometry_shader_text, NULL);            check(__FILE__, __LINE__);
    glCompileShader(geometry_shader);                                           check(__FILE__, __LINE__);
    glGetShaderiv (geometry_shader, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) print_shader_info_log(geometry_shader) ; 

    GLuint fragment_shader ; 
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);                       check(__FILE__, __LINE__);
    glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);            check(__FILE__, __LINE__);
    glCompileShader(fragment_shader);                                           check(__FILE__, __LINE__);
    glGetShaderiv (fragment_shader, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) print_shader_info_log(fragment_shader) ; 

    GLuint program;
    program = glCreateProgram();               check(__FILE__, __LINE__);
    glAttachShader(program, vertex_shader);    check(__FILE__, __LINE__);
    glAttachShader(program, geometry_shader);  check(__FILE__, __LINE__);
    glAttachShader(program, fragment_shader);  check(__FILE__, __LINE__);
    glLinkProgram(program);                    check(__FILE__, __LINE__);



    GLint LightPosition_location ; 
    GLint ModelViewProjection_location ; 
    GLint Param_location ; 
    LightPosition_location = glGetUniformLocation(program, "LightPosition");               check(__FILE__, __LINE__);
    ModelViewProjection_location = glGetUniformLocation(program, "ModelViewProjection");   check(__FILE__, __LINE__);
    Param_location = glGetUniformLocation(program, "Param");                               check(__FILE__, __LINE__);

    GLint vpos_location ; 
    GLint vdir_location ; 
    GLint vcol_location ; 
    vpos_location = glGetAttribLocation(program, "vpos");  check(__FILE__, __LINE__);
    vdir_location = glGetAttribLocation(program, "vdir");  check(__FILE__, __LINE__);
    vcol_location = glGetAttribLocation(program, "vcol");  check(__FILE__, __LINE__);

    std::cout << " vpos_location " << vpos_location << std::endl ; 
    std::cout << " vdir_location " << vdir_location << std::endl ; 
    std::cout << " vcol_location " << vcol_location << std::endl ; 


    glUseProgram(program);

    if(vpos_location > 0)  // it gets optimized away if not actually used in program as a whole ?
    {
        glEnableVertexAttribArray(vpos_location);                                                                     check(__FILE__, __LINE__);
        glVertexAttribPointer(vpos_location, 4, GL_FLOAT, GL_FALSE, sizeof(float)*4*3, (void*)(sizeof(float)*0));     check(__FILE__, __LINE__);
    }

    glEnableVertexAttribArray(vdir_location);                                                                     check(__FILE__, __LINE__);
    glVertexAttribPointer(vdir_location, 4, GL_FLOAT, GL_FALSE, sizeof(float)*4*3, (void*)(sizeof(float)*4));     check(__FILE__, __LINE__);

    glEnableVertexAttribArray(vcol_location);                                                                     check(__FILE__, __LINE__);
    glVertexAttribPointer(vcol_location, 4, GL_FLOAT, GL_FALSE, sizeof(float)*4*3, (void*)(sizeof(float)*8));     check(__FILE__, __LINE__);



    while (!glfwWindowShouldClose(window))
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glm::mat4 mvp = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
        glm::vec4 LightPosition(0.f, 0.f, 0.f, 1.0);
        glm::vec4 Param(100.f, 0.f, 0.f, 0.0);

        glUseProgram(program);
        glUniformMatrix4fv(ModelViewProjection_location, 1, GL_FALSE, (const GLfloat*) glm::value_ptr( mvp ) );
        glUniform4fv(LightPosition_location, 1, glm::value_ptr(LightPosition) );
        glUniform4fv(Param_location, 1, glm::value_ptr(Param) );

        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
