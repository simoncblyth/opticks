// http://antongerdelan.net/opengl/hellotriangle.html

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define INTEROP 1

#ifdef INTEROP
#include <cuda_gl_interop.h>
struct cudaGraphicsResource *vbo_cuda;

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#endif


#include <stdio.h>
#include "assert.h"

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

float points[] = {
   0.0f,  0.5f,  0.0f,
   0.5f, -0.5f,  0.0f,
  -0.5f, -0.5f,  0.0f
};

GLuint init_geometry()
{
    GLuint vbo ; 
    glGenBuffers (1, &vbo);
    glBindBuffer (GL_ARRAY_BUFFER, vbo);

#ifdef INTEROP
    GLenum usage = GL_DYNAMIC_DRAW ; 
    printf("DYNAMIC_DRAW\n");
#else
    GLenum usage = GL_STATIC_DRAW ; 
    printf("STATIC_DRAW\n");
#endif
    glBufferData (GL_ARRAY_BUFFER, 3 * 3 * sizeof (float), points, usage );
    return vbo ; 
}

GLuint init_buffer_description(GLuint vbo)
{
    GLuint vao ; 
    glGenVertexArrays (1, &vao);
    glBindVertexArray (vao);
    glEnableVertexAttribArray (0);
    glBindBuffer (GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    return vao ; 
}

const char* vertex_shader =
"#version 400\n"
"in vec3 vp;"
"void main () {"
"  gl_Position = vec4 (vp, 1.0);"
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



#ifdef INTEROP
struct grow
{
    unsigned int count ; 
    grow(unsigned int count) : count(count) {}

    __host__ __device__ float3 operator()(unsigned int i)
    {
         float s = 0.01f * (count % 100)  ; 
         float3 vec ; 
         switch(i)
         {
            case 0: vec = make_float3( 0.0f,     s,  0.0f) ; break ;
            case 1: vec = make_float3(    s,    -s,  0.0f) ; break ;
            case 2: vec = make_float3(   -s,    -s,  0.0f) ; break ;
         }  

         return vec ; 
    }
};

void init_interop(GLuint vbo)
{
    cudaGLSetGLDevice(0);
    cudaGraphicsGLRegisterBuffer(&vbo_cuda, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

void interop_move(unsigned int n)
{
    float3* raw_ptr;
    size_t buf_size;

    cudaGraphicsMapResources(1, &vbo_cuda, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&raw_ptr, &buf_size, vbo_cuda);

    //printf("interop_move buf_size:%lu \n", buf_size);

    thrust::device_ptr<float3> dev_ptr = thrust::device_pointer_cast(raw_ptr);

    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last(3);
    thrust::transform(first, last, dev_ptr,  grow(n) );

    cudaGraphicsUnmapResources(1, &vbo_cuda, 0);
}
#endif



int main () 
{
    init_glfw();
    init_gl();                                  

    GLuint vbo = init_geometry();
    GLuint vao = init_buffer_description(vbo);
    GLuint shader_program = init_shader();

#ifdef INTEROP
    init_interop(vbo);
    unsigned int n(0);
#endif

    while (!glfwWindowShouldClose (window)) 
    {
          glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

          glUseProgram (shader_program);
          glBindVertexArray (vao);
          glDrawArrays (GL_LINE_LOOP, 0, 3);

#ifdef INTEROP
          interop_move(n++);
#endif

          glfwPollEvents ();
          glfwSwapBuffers (window);
    }
 
    glfwTerminate();
    return 0;
}




