// WITH_GL the thrust scaling does happen but it is not seen by OpenGL
#define WITH_GL 1

#include <stdio.h>
#include "assert.h"

#ifdef WITH_GL
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#endif
#include "gloptixthrust.hh"

#ifdef WITH_GL
GLFWwindow* window = NULL ; 
void init_glfw()
{
    assert(glfwInit());
#ifdef __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
    window = glfwCreateWindow (640, 480, "Hello Triangle", NULL, NULL);
    glfwMakeContextCurrent (window);
}

void init_gl()
{
    glewExperimental = GL_TRUE;
    glewInit ();

    const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString (GL_VERSION); // version as a string
    printf ("Renderer: %s\n", renderer);
    printf ("OpenGL version supported %s\n", version);

    // tell GL to only draw onto a pixel if the shape is closer to the viewer
    glEnable (GL_DEPTH_TEST); // enable depth-testing
    glDepthFunc (GL_LESS);    // depth-testing interprets a smaller value as "closer"
}


GLuint init_vbo(unsigned int nvert)
{
    GLuint vbo ; 
    glGenBuffers (1, &vbo);
    glBindBuffer (GL_ARRAY_BUFFER, vbo);

    //GLenum usage = GL_DYNAMIC_DRAW ; 
    GLenum usage = GL_STREAM_DRAW ; 
    switch(usage)
    {
      case GL_DYNAMIC_DRAW: printf("DYNAMIC_DRAW\n");break;
      case GL_STREAM_DRAW:  printf("STREAM_DRAW\n");break;
    }
    
    glBufferData (GL_ARRAY_BUFFER, nvert * 4 * sizeof (float), NULL, usage );
    return vbo ; 
}

GLuint init_vao(GLuint vbo)
{
    GLuint vao ; 
    glGenVertexArrays (1, &vao);
    glBindVertexArray (vao);
    glEnableVertexAttribArray (0);
    glBindBuffer (GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer (0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    return vao ; 
}

const char* vertex_shader =
"#version 400\n"
"in vec4 vp;"
"void main () {"
"  gl_Position = vec4 (vec3(vp), 1.0);"
"}";

const char* fragment_shader =
"#version 400\n"
"out vec4 frag_colour;"
"void main () {"
"  frag_colour = vec4 (1.0, 1.0, 1.0, 1.0);"
"}";


GLuint init_shader()
{
    GLuint shader_program = glCreateProgram ();
    GLuint vs = glCreateShader (GL_VERTEX_SHADER);
    glShaderSource (vs, 1, &vertex_shader, NULL);
    glCompileShader (vs);

    GLuint fs = glCreateShader (GL_FRAGMENT_SHADER);
    glShaderSource (fs, 1, &fragment_shader, NULL);
    glCompileShader (fs);

    glAttachShader (shader_program, fs);
    glAttachShader (shader_program, vs);

    glLinkProgram (shader_program);
    return shader_program ;
}
#endif


int main () 
{
    unsigned int nvert = 10 ; 

#ifdef WITH_GL

    cudaGLSetGLDevice(0);

    init_glfw();
    init_gl();                                 

    GLuint vbo = init_vbo(nvert);
    GLuint vao = init_vao(vbo);
    GLuint shader = init_shader();

    GLOptiXThrust glot(vbo, nvert);
#else
    GLOptiXThrust glot(0, nvert);
#endif
    glot.compile();
    glot.launch(GLOptiXThrust::raygen_minimal_entry); // generate vertices

    float factor = 0.1f ; 
    glot.postprocess(factor);  // scale vertices : not reflected in drawing by OpenGL
    glot.sync();

    glot.launch(GLOptiXThrust::raygen_dump_entry);   


#ifdef WITH_GL
    while (!glfwWindowShouldClose (window)) 
    {
          glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

          glUseProgram (shader);
          glBindVertexArray (vao);
          glDrawArrays (GL_LINE_LOOP, 0, nvert);

          glfwPollEvents ();
          glfwSwapBuffers (window);
    }
    glfwTerminate();
#endif

    return 0;
}




