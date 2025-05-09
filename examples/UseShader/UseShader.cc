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


// 2d position and color together

static const struct
{
    float x, y;
    float r, g, b;
} vertices[3] =
{
    { -0.6f, -0.4f, 1.f, 0.f, 0.f },
    {  0.6f, -0.4f, 0.f, 1.f, 0.f },
    {   0.f,  0.6f, 0.f, 0.f, 1.f }
};



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


    if(!ok) std::cout << "OpenGL ERROR " << path << ":" << line << ":" << std::hex << err << std::dec << s << std::endl ; 
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


// on macOS with 
//    #version 460 core
// get runtime error, unsupported version
//
// on macOS with 
//    #version 410 core
// note that a trailing semicolon after the main curly brackets 
// gives a syntax error, that did not see on Linux with "#version 460 core"
//

static const char* vertex_shader_text = R"glsl(
#version 410 core
uniform mat4 MVP;
in vec3 vCol;
in vec2 vPos;
out vec3 color;
void main()
{
    gl_Position = MVP * vec4(vPos, 0.0, 1.0);
    color = vCol;
}

)glsl";


/**
    color = vCol;
    color = vec3(1.0,0.0,0.0);
**/


static const char* fragment_shader_text = R"glsl(
#version 410 core
in vec3 color;
out vec4 frag_color;

void main()
{
    frag_color = vec4(color, 1.0);
}

)glsl";


/**
    frag_color = vec4(1.0,0.0,0.0, 1.0);

**/



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

    //[ SGLFW::init

    GLFWwindow* window;


    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        exit(EXIT_FAILURE);

/*
* version specifies the minimum
*/

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
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // remove stuff deprecated in requested release
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint( GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);   // https://learnopengl.com/In-Practice/Debugging Debug output is core since OpenGL version 4.3,   
    
    // Frame::gl_init_window Renderer: TITAN RTX/PCIe/SSE2
    // Frame::gl_init_window OpenGL version supported 4.1.0 NVIDIA 418.56
#endif


    //window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
    window = glfwCreateWindow(1024, 768, "Simple example", NULL, NULL);
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
    assert( glGetError() == GL_INVALID_ENUM );  
    assert( glGetError() == GL_NO_ERROR );  

    const GLubyte* renderer = glGetString (GL_RENDERER);
    const GLubyte* version = glGetString (GL_VERSION); 
    printf(" renderer %s \n", renderer ); 
    printf(" version %s \n", version ); 

    glfwSwapInterval(1);

    //] SGLFW::init


    unsigned vao ;                                                              check(__FILE__, __LINE__);
    glGenVertexArrays (1, &vao);                                                check(__FILE__, __LINE__);
    glBindVertexArray (vao);                                                    check(__FILE__, __LINE__);
    
    GLuint vertex_buffer ;
    glGenBuffers(1, &vertex_buffer);                                            check(__FILE__, __LINE__);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);                               check(__FILE__, __LINE__);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);  check(__FILE__, __LINE__);
    // vec2 position and vec3 color within the vertices buffer


    GLuint program;
    { 
        // compile vertex shader
        GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);                    check(__FILE__, __LINE__);
        glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);                check(__FILE__, __LINE__);
        glCompileShader(vertex_shader);                                             check(__FILE__, __LINE__);
        int v_params = -1; 
        glGetShaderiv (vertex_shader, GL_COMPILE_STATUS, &v_params);
        if (GL_TRUE != v_params) print_shader_info_log(vertex_shader) ; 

        // compile fragment shader
        GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);                check(__FILE__, __LINE__);
        glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);            check(__FILE__, __LINE__);
        glCompileShader(fragment_shader);                                           check(__FILE__, __LINE__);
        int f_params = -1; 
        glGetShaderiv (fragment_shader, GL_COMPILE_STATUS, &f_params);
        if (GL_TRUE != f_params) print_shader_info_log(fragment_shader) ; 

        // link program
        program = glCreateProgram();               check(__FILE__, __LINE__);
        glAttachShader(program, vertex_shader);    check(__FILE__, __LINE__);
        glAttachShader(program, fragment_shader);  check(__FILE__, __LINE__);
        glLinkProgram(program);                    check(__FILE__, __LINE__);
    }


    // get uniform and attribute locations from program
    GLint mvp_location = glGetUniformLocation(program, "MVP");   check(__FILE__, __LINE__);
    GLint vpos_location = glGetAttribLocation(program, "vPos");  check(__FILE__, __LINE__);
    GLint vcol_location = glGetAttribLocation(program, "vCol");  check(__FILE__, __LINE__);

    printf("//mvp_location  %d \n", mvp_location );  
    printf("//vpos_location %d \n", vpos_location );  
    printf("//vcol_location %d \n", vcol_location );  

    glEnableVertexAttribArray(vpos_location);                                  check(__FILE__, __LINE__);
    glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*) 0);                    check(__FILE__, __LINE__);
    glEnableVertexAttribArray(vcol_location);                                                                     check(__FILE__, __LINE__);
    glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*) (sizeof(float) * 2));  check(__FILE__, __LINE__);
    //  the offset of 2 floats picks the color triplet from the single vertices buffer


    int count(0);
    int renderlooplimit(200); 
    bool exitloop(false); 

    while (!glfwWindowShouldClose(window) && !exitloop)
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glm::mat4 mvp = glm::scale(glm::mat4(1.0f), glm::vec3(1.5f));
        float* mvp_f = glm::value_ptr( mvp ); 

        glUseProgram(program);
        glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (const GLfloat*) mvp_f );
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(window);
        glfwPollEvents();

        exitloop = renderlooplimit > 0 && count++ > renderlooplimit ;   
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
