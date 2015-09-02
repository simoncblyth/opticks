// http://antongerdelan.net/opengl/hellotriangle.html

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define INTEROP 1 
#ifdef INTEROP
#include "CudaGLBuffer.hh"
#include "callgrow.hh"
#include <optixu/optixpp_namespace.h>

enum { raygen_entry, num_entry } ;
std::string ptxpath(const char* name, const char* ptxdir){
    char path[128] ; 
    snprintf(path, 128, "%s/%s", getenv(ptxdir), name );
    return path ;   
}
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


GLuint init_geometry()
{
    GLuint vbo ; 
    glGenBuffers (1, &vbo);
    glBindBuffer (GL_ARRAY_BUFFER, vbo);

    GLenum usage = GL_DYNAMIC_DRAW ; 
    printf("DYNAMIC_DRAW\n");
    glBufferData (GL_ARRAY_BUFFER, 3 * 3 * sizeof (float), NULL, usage );
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


int main () 
{
    init_glfw();
    init_gl();                                 

    GLuint vbo = init_geometry();
    GLuint vao = init_buffer_description(vbo);
    GLuint shader_program = init_shader();


#ifdef INTEROP
    cudaGLSetGLDevice(0);

    optix::Context context = optix::Context::create();
    context->setPrintEnabled(true);
    context->setPrintBufferSize(8192);
    context->setStackSize( 2180 );

    context->setEntryPointCount(num_entry);

    optix::Program raygen = context->createProgramFromPTXFile( ptxpath("cgb.ptx", "PTXDIR"), "cgb" );

    context->setRayGenerationProgram( raygen_entry, raygen );

    std::cout << "validate " << std::endl ; 

    context->validate();

    CudaGLBuffer<float3>* cgb = new CudaGLBuffer<float3>(vbo, cudaGraphicsMapFlagsWriteDiscard, 0);

    cgb->mapResources();
    {
        float3*  d_ptr = cgb->getRawPtr() ;

        unsigned int count = cgb->getCount() ; 

        optix::Buffer buffer = context->createBufferForCUDA(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, count);

        CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(d_ptr) ; 

        unsigned int device_number = 0u ; 

        buffer->setDevicePointer(device_number, cu_ptr );

        context["cgb_buffer"]->set( buffer );   // cannot compile context without this

        context->compile();

        context->launch(raygen_entry, 0);

        context->launch(raygen_entry, count);

        cgb->Summary();

    }
    cgb->unmapResources();


    unsigned int n(0);
#endif


    while (!glfwWindowShouldClose (window)) 
    {
          glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

          glUseProgram (shader_program);
          glBindVertexArray (vao);

#ifdef INTEROP
          callgrow( cgb, n );
          n++ ; 
#endif

          glDrawArrays (GL_LINE_LOOP, 0, 3);

          glfwPollEvents ();
          glfwSwapBuffers (window);
    }
 
    glfwTerminate();
    return 0;
}


