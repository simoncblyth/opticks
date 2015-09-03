// http://antongerdelan.net/opengl/hellotriangle.html

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define INTEROP 1 
#ifdef INTEROP
#include "CudaGLBuffer.hh"
#include "callgrow.hh"
//#include "grow.hh"
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


GLuint init_geometry(unsigned int nvert)
{
    GLuint vbo ; 
    glGenBuffers (1, &vbo);
    glBindBuffer (GL_ARRAY_BUFFER, vbo);

    GLenum usage = GL_DYNAMIC_DRAW ; 
    printf("DYNAMIC_DRAW\n");
    glBufferData (GL_ARRAY_BUFFER, nvert * 3 * sizeof (float), NULL, usage );
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

    //unsigned int nvert = 1000 ; 
    unsigned int nvert = 4 ; 
    GLuint vbo = init_geometry(nvert);
    GLuint vao = init_buffer_description(vbo);
    GLuint shader_program = init_shader();


#ifdef INTEROP
    cudaGLSetGLDevice(0);

    //cudaGraphicsMapFlagsNone         //Default; Assume resource can be read/written
    //cudaGraphicsMapFlagsReadOnly     //CUDA will not write to this resource
    //cudaGraphicsMapFlagsWriteDiscard //CUDA will only write to and will not read from this resource

    unsigned int flags = cudaGraphicsMapFlagsNone ; // Default; Assume resource can be read/written 

    // grabbing the OpenGL buffer for use by CUDA
    CudaGLBuffer<float3>* cgb = new CudaGLBuffer<float3>(vbo, flags, 0);
    cgb->mapResources();    
    float3*  d_ptr = cgb->getRawPtr() ;
    unsigned int count = cgb->getCount() ; 
    cgb->Summary();


    bool optix = true ; 
    bool thrust = true ; 

    // both can write to the buffer, 
    // but it seems that thrust does not see what optix wrote

    if(optix)
    {
        optix::Context context = optix::Context::create();
        context->setPrintEnabled(true);
        context->setPrintBufferSize(8192);
        context->setStackSize( 2180 );

        context->setEntryPointCount(num_entry);
        optix::Program raygen = context->createProgramFromPTXFile( ptxpath("cgb.ptx", "PTXDIR"), "cgb" );
        context->setRayGenerationProgram( raygen_entry, raygen );
        std::cout << "validate " << std::endl ; 
        context->validate();

        // "createBufferForCUDA" seems misleading name as this creates a buffer **from** a CUdeviceptr
        // also remember that INPUT/OUTPUT for OptiX if from the GPU perspective

        optix::Buffer buffer = context->createBufferForCUDA(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, count);
        CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(d_ptr) ; 
        unsigned int device_number = 0u ; 
        buffer->setDevicePointer(device_number, cu_ptr );
        context["cgb_buffer"]->set( buffer );   // cannot compile context without this
        context->compile();
        context->launch(raygen_entry, 0);
        context->launch(raygen_entry, count);

        //buffer->markDirty();  // seems non-sensical but give it a go 

    }
    //cgb->unmapResources();


    if(thrust)
    {
        callgrow_value( cgb, 0 , false );  // this attempts to transform the preexisting content of buffer : not working 
        //callgrow_index( cgb, 0 , false );  // this just writes based in the index : works 
    }

    // https://devtalk.nvidia.com/default/topic/570952/optix/rtbuffersetdevicepointer-problem-to-update/


    // https://devtalk.nvidia.com/default/topic/558491
    //
    //      RT_BUFFER_INPUT - Only the host may write to the buffer. 
    //                        Data is transferred from host to device and device access is restricted to be read-only.
    //
    //                        if you have input only buffers, the data from the device is never copied back to the host. 
    //                        You can map the host buffer, but you will only get what you put in it last. 
    //
    //
    //      Marking something dirty simply tells OptiX that you changed the data on the device outside of OptiX,
    //



    unsigned int n(0);
#endif


    while (!glfwWindowShouldClose (window)) 
    {
          glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

          glUseProgram (shader_program);
          glBindVertexArray (vao);

#ifdef INTEROP

          if(thrust)
          { 
              callgrow_value( cgb, n , false );
              //callgrow_index( cgb, n , true );
              n++ ;
          } 
#endif

          glDrawArrays (GL_LINE_LOOP, 0, nvert);

          glfwPollEvents ();
          glfwSwapBuffers (window);
    }
 
    glfwTerminate();
    return 0;
}


