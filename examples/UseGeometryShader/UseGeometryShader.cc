/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
#include <glm/gtx/string_cast.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

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


std::string to_string(const glm::vec3& v)
{
    std::stringstream ss ; 
    for (int i=0; i<3; i++) ss << std::fixed << std::setprecision(3) << std::setw(10) << v[i] ;
    ss << std::endl ;
    return ss.str(); 
}
std::string to_string(const glm::vec4& v)
{
    std::stringstream ss ; 
    for (int i=0; i<4; i++) ss << std::fixed << std::setprecision(3) << std::setw(10) << v[i] ;
    ss << std::endl ;
    return ss.str(); 
}
std::string to_string(const glm::mat4& m)
{
    std::stringstream ss ; 
    for (int j=0; j<4; j++)
    {   
        for (int i=0; i<4; i++) ss << std::fixed << std::setprecision(3) << std::setw(10) << m[i][j] ;
        ss << std::endl ;
    }   
    return ss.str(); 
}

glm::mat4 getMVP(int width, int height, bool verbose)
{
    // Composition::setCenterExtent 
    glm::vec4 ce(0.f, 0.f, 0.f, 1000.f );  // center extent of the "model"

    glm::vec3 tr(ce.x, ce.y, ce.z);
    glm::vec3 sc(ce.w);
    glm::vec3 isc(1.f/ce.w);

    glm::mat4 model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
    glm::mat4 world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr); 

    // View::getTransforms
    glm::vec4 eye_m( -1.f,-1.f,0.f,1.f);   //  viewpoint in unit model frame 
    glm::vec4 look_m( 0.f, 0.f,0.f,1.f); 
    glm::vec4 up_m(   0.f, 0.f,1.f,1.f); 
    glm::vec4 gze_m( look_m - eye_m ) ; 

    const glm::mat4& m2w = model2world ; 
    glm::vec3 eye = glm::vec3( m2w * eye_m ) ; 
    glm::vec3 look = glm::vec3( m2w * look_m ) ; 
    glm::vec3 up = glm::vec3( m2w * up_m ) ; 
    glm::vec3 gaze = glm::vec3( m2w * gze_m ) ;    


    if(verbose)
    {
       std::cout << " model2world \n" << to_string( model2world ) << std::endl ; 
       std::cout << " world2model \n" << to_string( world2model ) << std::endl ; 
       std::cout << " eye \n" << to_string( eye ) << std::endl ; 
       std::cout << " look \n" << to_string( look ) << std::endl ; 
       std::cout << " up \n" << to_string( up ) << std::endl ; 
       std::cout << " gaze \n" << to_string( gaze ) << std::endl ; 
    }


    glm::vec3 forward_ax = glm::normalize(gaze);
    glm::vec3 right_ax   = glm::normalize(glm::cross(forward_ax,up)); 
    glm::vec3 top_ax     = glm::normalize(glm::cross(right_ax,forward_ax));

    // OpenGL eye space convention : -Z is forward, +X to right, +Y up
    glm::mat4 rot(1.0) ; 
    rot[0] = glm::vec4( right_ax, 0.f );  
    rot[1] = glm::vec4( top_ax  , 0.f );  
    rot[2] = glm::vec4( -forward_ax, 0.f );  

    glm::mat4 ti(glm::translate(glm::vec3(eye)));
    glm::mat4 t(glm::translate(glm::vec3(-eye)));  // eye to origin 

    float gazelength = glm::length(gaze);
    glm::mat4 eye2look = glm::translate( glm::mat4(1.), glm::vec3(0,0,gazelength));
    glm::mat4 look2eye = glm::translate( glm::mat4(1.), glm::vec3(0,0,-gazelength));

    glm::mat4 world2camera = glm::transpose(rot) * t  ;   
    glm::mat4 camera2world = ti * rot ;                   

    //glm::vec4 gaze = glm::vec4( gze, 0.f );

    // Composition::update
    glm::mat4 world2eye = world2camera ; // no look rotation or trackballing  

    float aspect = float(width)/float(height) ; 

    // Camera::aim
    float basis = 1000.f ;  
    float near = basis/10.f ; 
    float far = basis*5.f ; 
    float zoom = 1.0f ; 

    // Camera::getFrustum
    bool parallel = false ; 
    float orthoscale = 1.0f ; 
    float scale = parallel ? orthoscale : near   ; 
   
    float left = -aspect*scale/zoom ; 
    float right = aspect*scale/zoom ; 

    float top = scale/zoom ; 
    float bottom = -scale/zoom ; 
   
    glm::mat4 projection = glm::frustum( left, right, bottom, top, near, far ); 
    glm::mat4 world2clip = projection * world2eye ;    //  ModelViewProjection 


    if(verbose)
    {
       std::cout << " rot  \n" << to_string( rot ) << std::endl ; 
       std::cout << " eye2look  \n" << to_string( eye2look ) << std::endl ; 
       std::cout << " look2eye  \n" << to_string( look2eye ) << std::endl ; 
       std::cout << " world2camera \n" << to_string( world2camera ) << std::endl ; 
       std::cout << " camera2world \n" << to_string( camera2world ) << std::endl ; 
       std::cout << " projection \n" << to_string( projection ) << std::endl ; 
       std::cout << " world2clip \n" << to_string( world2clip ) << std::endl ; 
    }


    return world2clip ; 
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
        glm::vec4( 0.f   ,    0.f,    0.f,   0.f ), 
        glm::vec4( 1000.f,    0.f,    0.f,   0.f ), 
        glm::vec4( 1.f   ,    0.f,    0.f,   0.f ), 

        glm::vec4( 0.f   ,    0.f,    0.f,   0.f ), 
        glm::vec4( 0.f   , 1000.f,    0.f,   0.f ), 
        glm::vec4( 0.f   ,    1.f,    0.f,   0.f ), 

        glm::vec4( 0.f   ,    0.f,    0.f,   0.f ),
        glm::vec4( 0.f   ,    0.f, 1000.f,   0.f ),
        glm::vec4( 0.f   ,    0.f,    1.f,   0.f )
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
    geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);                       check(__FILE__, __LINE__);
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


    bool verbose(true) ; 
    bool exitloop(false);
    int count(0); 
    int renderlooplimit(200);

    while (!glfwWindowShouldClose(window) && !exitloop)
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glm::mat4 world2clip = getMVP(width, height, verbose) ; 
        verbose = false ; 

        glm::vec4 LightPosition(0.f, 0.f, 0.f, 1.0);
        glm::vec4 Param(1.f, 0.f, 0.f, 0.0);

        glUseProgram(program);
        glUniformMatrix4fv(ModelViewProjection_location, 1, GL_FALSE, (const GLfloat*) glm::value_ptr( world2clip ) );
        glUniform4fv(LightPosition_location, 1, glm::value_ptr(LightPosition) );
        glUniform4fv(Param_location, 1, glm::value_ptr(Param) );

        glDrawArrays(GL_POINTS, 0, 3);
        glfwSwapBuffers(window);
        glfwPollEvents();
      
        exitloop = renderlooplimit > 0 && count > renderlooplimit ; 
        count++ ; 
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}


















