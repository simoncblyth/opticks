/**
UseGeometryShader : flying point visualization of uncompressed step-by-step photon records
=============================================================================================

* started from https://www.glfw.org/docs/latest/quick.html#quick_example
* reference on geometry shaders https://open.gl/geometry
* see notes/issues/geometry-shader-flying-photon-visualization.rst

TODO: more encapsulation/centralization of GLFW/OpenGL mechanics and viz math down 
      into the header only imps: SGLFW.hh and SGLM.hh

**/

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include "SGLFW.hh"
#include "SGLM.hh"
#include "NP.hh"

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

    NP* record = NP::Load(FOLD, "r.npy") ;   // expecting shape like (10000, 10, 4, 4)
    std::cout <<  " record " << record->sstr() << std::endl ; 
    GLint   record_first = 0 ; 
    GLsizei record_count = record->shape[0]*record->shape[1] ;  


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
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // remove stuff deprecated in requested release
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint( GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);   // https://learnopengl.com/In-Practice/Debugging Debug output is core since OpenGL version 4.3,   
#endif

    GLFWwindow* window;
    window = glfwCreateWindow(1280, 720, "UseGeometryShader", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwSetKeyCallback(window, SGLFW::key_callback);
    glfwMakeContextCurrent(window);
    
    glewExperimental = GL_TRUE;
    glewInit (); 
    assert( glGetError() == GL_INVALID_ENUM );  // long-standing glew bug apparently 
    assert( glGetError() == GL_NO_ERROR );  

    const GLubyte* renderer = glGetString (GL_RENDERER);
    const GLubyte* version = glGetString (GL_VERSION); 
    printf(" renderer %s \n", renderer ); 
    printf(" version %s \n", version ); 

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE); // otherwise gl_PointSize setting ignored, setting in geom not vert shader used when present 

    glfwSwapInterval(1);
    unsigned vao ;                                                              SGLFW::check(__FILE__, __LINE__); 
    glGenVertexArrays (1, &vao);                                                SGLFW::check(__FILE__, __LINE__);
    glBindVertexArray (vao);                                                    SGLFW::check(__FILE__, __LINE__);

    GLuint vertex_buffer ; 
    glGenBuffers(1, &vertex_buffer);                                            SGLFW::check(__FILE__, __LINE__);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);                               SGLFW::check(__FILE__, __LINE__);
    glBufferData(GL_ARRAY_BUFFER, record->arr_bytes(), record->bytes(), GL_STATIC_DRAW);  SGLFW::check(__FILE__, __LINE__);


    int params = -1; 
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);                    SGLFW::check(__FILE__, __LINE__);
    glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);                SGLFW::check(__FILE__, __LINE__);
    glCompileShader(vertex_shader);                                             SGLFW::check(__FILE__, __LINE__);
    glGetShaderiv (vertex_shader, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) SGLFW::print_shader_info_log(vertex_shader) ; 

    GLuint geometry_shader = 0 ;
    if( geometry_shader_text )
    {
        geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);                       SGLFW::check(__FILE__, __LINE__);
        glShaderSource(geometry_shader, 1, &geometry_shader_text, NULL);            SGLFW::check(__FILE__, __LINE__);
        glCompileShader(geometry_shader);                                           SGLFW::check(__FILE__, __LINE__);
        glGetShaderiv (geometry_shader, GL_COMPILE_STATUS, &params);
        if (GL_TRUE != params) SGLFW::print_shader_info_log(geometry_shader) ; 
    }

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);                SGLFW::check(__FILE__, __LINE__);
    glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);            SGLFW::check(__FILE__, __LINE__);
    glCompileShader(fragment_shader);                                           SGLFW::check(__FILE__, __LINE__);
    glGetShaderiv (fragment_shader, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) SGLFW::print_shader_info_log(fragment_shader) ; 

    GLuint program = glCreateProgram();        SGLFW::check(__FILE__, __LINE__);
    glAttachShader(program, vertex_shader);    SGLFW::check(__FILE__, __LINE__);
    if( geometry_shader > 0 ) glAttachShader(program, geometry_shader);  SGLFW::check(__FILE__, __LINE__);
    glAttachShader(program, fragment_shader);  SGLFW::check(__FILE__, __LINE__);
    glLinkProgram(program);                    SGLFW::check(__FILE__, __LINE__);



    GLint ModelViewProjection_location = glGetUniformLocation(program, "ModelViewProjection");   SGLFW::check(__FILE__, __LINE__);
    GLint Param_location               = glGetUniformLocation(program, "Param");                 SGLFW::check(__FILE__, __LINE__);

    GLint rpos_location = glGetAttribLocation(program, "rpos");  SGLFW::check(__FILE__, __LINE__);
    std::cout << " rpos_location " << rpos_location << std::endl ; 

    glUseProgram(program);

    GLsizei stride = sizeof(float)*4*4 ;  
    const void* rpos_offset = (void*)(sizeof(float)*0) ;   // pos

    if( rpos_location > -1 )
    { 
        glEnableVertexAttribArray(rpos_location);                                              SGLFW::check(__FILE__, __LINE__);
        glVertexAttribPointer(rpos_location, 4, GL_FLOAT, GL_FALSE, stride, rpos_offset );     SGLFW::check(__FILE__, __LINE__);
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    std::cout << " width " << width << " height " << height << std::endl ; 

 
    SGLM sglm ; 
    sglm.width = width ; 
    sglm.height = height ; 
    sglm.zoom = 1.f ;   
    sglm.eye_m.x = -2.f ; 
    sglm.eye_m.y = -2.f ; 
    sglm.eye_m.z = 0.f ; 
    sglm.setExtent(50.f); 
    sglm.update(); 
    sglm.dump(); 

    const glm::mat4& world2clip = sglm.world2clip ; 
    const GLfloat* mvp = (const GLfloat*) glm::value_ptr(world2clip) ;  

    glm::vec4 Param(0.f, 1.f, 0.001f, 0.f); // t0, t1, dt, tc 

    bool exitloop(false);
    int renderlooplimit(2000);
    int count(0); 

    while (!glfwWindowShouldClose(window) && !exitloop)
    {
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        Param.w += Param.z ; if( Param.w > Param.y ) Param.w = Param.x ; 

        glUseProgram(program);
        glUniformMatrix4fv(ModelViewProjection_location, 1, GL_FALSE, mvp );
        glUniform4fv(      Param_location,               1, glm::value_ptr(Param) );

        glDrawArrays(GL_LINE_STRIP, record_first, record_count);
        glfwSwapBuffers(window);
        glfwPollEvents();
      
        exitloop = renderlooplimit > 0 && count > renderlooplimit ; 
        count++ ; 
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

