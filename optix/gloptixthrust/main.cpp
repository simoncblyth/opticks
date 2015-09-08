#include <stdio.h>
#include "assert.h"
#include <unistd.h>
#include <algorithm>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "OBuffer.hh"
#include "CResource.hh"
#include "TAry.hh"
#include "TProc.hh"


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


struct VBO {
      unsigned int id ;
      unsigned int target  ;
      unsigned int usage ;
      void*        data ;
      unsigned int num_bytes ; 

      VBO(GLenum target, GLenum usage) : target(target), usage(usage), data(NULL), num_bytes(0)
      {
          glGenBuffers (1, &id);
      }
      void upload(void* _data, unsigned int _num_bytes)
      {
          printf("VBO::upload %p %u \n", _data, _num_bytes );  
          glBindBuffer(target, id) ;
          glBufferData(target, _num_bytes, _data, usage );
          glBindBuffer(target, 0) ;

          data = _data ; 
          num_bytes = _num_bytes;
      }
      void bind()
      {
          glBindBuffer (target, id); 
      }
      void unbind()
      {
          glBindBuffer (target, 0); 
      }


      void Summary(const char* msg)
      {
          printf("%s\n", msg);
          switch(usage)
          {
              case GL_STATIC_DRAW:  printf("STATIC_DRAW\n");break;
              case GL_DYNAMIC_DRAW: printf("DYNAMIC_DRAW\n");break;
              case GL_STREAM_DRAW:  printf("STREAM_DRAW\n");break;
           }
      } 
};

class App {
    public:
       static const char* CMAKE_TARGET ; 
       static const char* vertex_shader ; 
       static const char* fragment_shader ; 
    public:
       enum { e_vtx, e_sel, e_seq } ;
       enum { raygen_circle_entry, raygen_dump_entry, num_entry } ;
    public:
       App(unsigned int nvert);
    private:
       void init();
       void init_vao();
       void init_shader();
       void init_gl_buffers();
       void init_optix();
       void init_buffers();
       static float* make_triangle_data(float s);
       static unsigned int* make_uvec4_data(unsigned int nvert, unsigned int value);
       void addRayGenerationProgram( const char* ptxname, const char* progname, unsigned int entry );
    public:
       void generate(float radius);
       void tgenerate(float radius);
       void tscale(float factor);
       void index();
       void render();
       void sleep(unsigned int microseconds);
       void dump_seq(const char* msg);
       void dump_sel(const char* msg);
       template <typename T> void mapped_dump(unsigned int id);
    private:
       unsigned int m_device ;  
    private:
       // OpenGL names
       unsigned int  m_vao ;  
       VBO*          m_vtx ;  
       VBO*          m_seq ;  
       VBO*          m_sel ;  
       unsigned int  m_shader ; 
    private:
       unsigned int m_nvert ;  
    private:
       optix::Context  m_context ; //    
       OBuffer*        m_vtxbuf ;  // 
       OBuffer*        m_seqbuf ;  //   
       OBuffer*        m_selbuf ; 
};






inline App::App(unsigned int nvert) :
   m_device(0),
   m_vao(0),
   m_vtx(NULL),
   m_seq(NULL),
   m_sel(NULL),
   m_shader(0),
   m_nvert(nvert),
   m_vtxbuf(NULL),
   m_seqbuf(NULL),
   m_selbuf(NULL)
{
   init();
}  

void App::init()
{
    init_gl_buffers();
    init_vao();
    init_shader();
    init_optix();  
    init_buffers();
}


float* App::make_triangle_data(float s)
{
    float* data = new float[3*4] ;
    for(unsigned int i=0 ; i < 3 ; i++)
    {
        switch(i)
        {
            case 0:
                   data[4*i+0] = -s ; 
                   data[4*i+1] = -s ; 
                   data[4*i+2] =  0.f ; 
                   data[4*i+3] =  1.f ;
                   break ; 
            case 1:
                   data[4*i+0] =  s ; 
                   data[4*i+1] = -s ; 
                   data[4*i+2] =  0.f ; 
                   data[4*i+3] =  1.f ;
                   break ; 
            case 2:
                   data[4*i+0] =  0.f ; 
                   data[4*i+1] =  s   ; 
                   data[4*i+2] =  0.f ; 
                   data[4*i+3] =  1.f ;
                   break ; 
        }
    }
    return data ; 
}

unsigned int* App::make_uvec4_data(unsigned int nvert, unsigned int value)
{
    unsigned int* data = new unsigned int[nvert*4] ;
    for(unsigned int i=0 ; i < nvert ; i++)
    {
        data[i*4+0] = value ; 
        data[i*4+1] = value ; 
        data[i*4+2] = value ; 
        data[i*4+3] = value ; 
    }
    return data ; 
}


void App::init_vao()
{
    glGenVertexArrays (1, &m_vao);
    glBindVertexArray (m_vao);
    if(m_vtx)
    {
        printf("App::init_vao vtx %d \n", m_vtx->id);
        m_vtx->bind();
        glVertexAttribPointer (e_vtx, 4, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray (e_vtx);
        m_vtx->unbind();
    }
    if(m_sel)
    {
        printf("App::init_vao sel %d \n", m_sel->id);
        m_sel->bind();
        glVertexAttribIPointer (e_sel, 4, GL_UNSIGNED_INT, 0, NULL);  // glVertexAttrib*I*Pointer 
        glEnableVertexAttribArray (e_sel);
        m_sel->unbind();
    }
    if(m_seq)
    {
        printf("App::init_vao seq %d \n", m_seq->id);
        m_seq->bind();
        glVertexAttribIPointer (e_seq, 4, GL_UNSIGNED_INT, 0, NULL);  // glVertexAttrib*I*Pointer 
        glEnableVertexAttribArray (e_seq);
        m_seq->unbind();
    }



}


const char* App::vertex_shader =
"#version 400\n"
"in vec4 vp;"
"in uvec4 vsel;"
"out vec4 colour;"
"void main () {"
"  switch(vsel.x){ "
"    case 0: colour = vec4(1.0, 1.0, 1.0, 1.0) ; break ; "
"    case 1: colour = vec4(1.0, 0.0, 0.0, 1.0) ; break ; "
"    case 2: colour = vec4(0.0, 1.0, 0.0, 1.0) ; break ; "
"    case 3: colour = vec4(0.0, 0.0, 1.0, 1.0) ; break ; "
"  };"
"  gl_Position = vec4 (vec3(vp), 1.0);"
"}";


const char* App::fragment_shader =
"#version 400\n"
"in  vec4 colour;"
"out vec4 frag_colour;"
"void main () {"
"     frag_colour = colour;"
"}";

void App::init_shader()
{
    m_shader = glCreateProgram ();
    GLuint vs = glCreateShader (GL_VERTEX_SHADER);
    glShaderSource (vs, 1, &vertex_shader, NULL);
    glCompileShader (vs);

    GLuint fs = glCreateShader (GL_FRAGMENT_SHADER);
    glShaderSource (fs, 1, &fragment_shader, NULL);
    glCompileShader (fs);

    glAttachShader (m_shader, fs);
    glAttachShader (m_shader, vs);

    glLinkProgram (m_shader);

    printf("App::init_shader m_shader %d \n", m_shader);
}


const char* App::CMAKE_TARGET = "GLOptiXThrustMinimal" ; 

std::string ptxpath(const char* name, const char* cmake_target ){
    char path[128] ; 
    snprintf(path, 128, "%s/%s_generated_%s", getenv("PTXDIR"), cmake_target, name );
    return path ;  
}

void App::addRayGenerationProgram( const char* ptxname, const char* progname, unsigned int entry )
{
    std::string path = ptxpath(ptxname, CMAKE_TARGET) ;
    printf("App::addRayGenerationProgram (%d) %s %s : %s \n", entry, ptxname, progname, path.c_str());
    optix::Program prog = m_context->createProgramFromPTXFile( path, progname );
    m_context->setRayGenerationProgram( entry, prog );
}

void App::init_optix()
{
    if(m_nvert == 3) return ; // OpenGL debug with a triangle

    m_context = optix::Context::create();
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setStackSize( 2180 );

    m_context->setEntryPointCount(num_entry);

    addRayGenerationProgram("circle.cu.ptx", "circle_make_vertices", raygen_circle_entry );
    addRayGenerationProgram("circle.cu.ptx", "circle_dump",          raygen_dump_entry );
}

/*
Although 3-way OpenGL/OptiX/Thrust RW might be made to work it is preferable to keep things simple
devise pairwise interop RW pairings and buffer interactions for multiple buffers 
to achieve the desired manipulations 

             OpenGL  OptiX  Thrust     notes  
   vtxbuf      CR      W      -        created by OpenGL, populated by OptiX, read by OpenGL for visualization
   seqbuf       -      W      R        created by OptiX, populated by OptiX, read by Thrust to create selbuf
   selbuf      CR      -      W        created by OpenGL, populated by Thrust, read by OpenGL for visualization of selections      

*/

void App::init_gl_buffers()
{
    m_vtx = new VBO(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    float* vtxdata = m_nvert == 3 ? make_triangle_data(0.9f) : NULL ; 
    m_vtx->upload( vtxdata,  m_nvert*4*sizeof(float) );

    m_seq = NULL ; 

    m_sel = new VBO(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    unsigned int* seldata = m_nvert == 3 ? make_uvec4_data(m_nvert, 2u) : NULL ; 
    m_sel->upload( seldata, m_nvert*4*sizeof(unsigned int) );  // still need to glBufferData even when nothing in it
}

void App::init_buffers()
{
    // bad API "too wide" : causing many args to be ignored
    m_vtxbuf = new OBuffer(m_context, m_vtx->id, "vtx_buffer", m_nvert, RT_FORMAT_FLOAT4,         RT_BUFFER_OUTPUT,       CResource::RW ); 
    m_seqbuf = new OBuffer(m_context, 0        , "seq_buffer", m_nvert, RT_FORMAT_UNSIGNED_INT4,  RT_BUFFER_OUTPUT,       CResource::RW ); 
    //     most of the seqbuf args are ignored as just need plain vanilla optix buffer
    m_selbuf = new OBuffer(m_context, m_sel->id, "sel_buffer", m_nvert, RT_FORMAT_UNSIGNED_INT4,  RT_BUFFER_INPUT_OUTPUT, CResource::W ); 
    //     most of the selbuf args are ignored as no OptiX buffer
    // unclear how to split though, common base ?
    //    all buffers have common features like size, num_bytes, basis type
    //
}

void App::generate(float radius)
{
    if(m_nvert == 3) return ;

    printf("App::generate %10.4f \n", radius);

    m_vtxbuf->map(OBuffer::GLToOptiX);  // createBufferFromGLBO
    m_seqbuf->map(OBuffer::OptiX);      // createBuffer

    m_context->validate();
    m_context->compile();
    m_context["radius"]->setFloat(radius) ; 

    printf("App::generate circle launch : 0 check \n");
    m_context->launch(raygen_circle_entry, 0 );      

    printf("App::generate circle launch : nvert %d \n", m_nvert );
    m_context->launch(raygen_circle_entry, m_nvert ); // generate vertices

    printf("App::generate dump launch : nvert %d \n", m_nvert );
    m_context->launch(raygen_dump_entry, m_nvert ); // dump vertices

    m_seqbuf->unmap();
    m_vtxbuf->unmap();
}

void App::tgenerate(float radius)
{
    if(m_nvert == 3) return ;
    printf("App::tgenerate \n");

    m_seqbuf->map(OBuffer::OptiX);      // createBuffer
    BufSpec vtx = m_vtxbuf->map(OBuffer::GLToCUDA);   // createBufferFromGLBO

    TProc tp(vtx);
    tp.tgenerate(radius);

    m_seqbuf->unmap(); 
    m_vtxbuf->unmap(); 
}

void App::tscale(float factor)
{
    if(m_nvert == 3) return ;
    //printf("App::tscale \n");

    BufSpec vtx = m_vtxbuf->map(OBuffer::GLToCUDA);  

    TProc tp(vtx);
    tp.tscale(factor);

    m_vtxbuf->unmap(); 
}


void App::index()
{
    if(m_nvert == 3) return ;
    printf("App::index \n");

    BufSpec seq = m_seqbuf->map(OBuffer::OptiXToCUDA);
    BufSpec sel = m_selbuf->map(OBuffer::GLToCUDA);  

    seq.Summary("seq OptiX created input");
    sel.Summary("sel GL created output ");

    TAry ta(seq, sel);
 
    //  CUDA funcs and kernel calls succeed to interop as expected  
    //ta.memcpy();
    //ta.memset();
    //ta.kcall();

    //  when avoid misunderstanding regards creating thrust::device_vector
    //  are now getting interop to work 
      ta.transform();  
    //ta.tcopy();
    //ta.tfill();

    m_seqbuf->unmap(); 

    //m_selbuf->streamSync();
    m_selbuf->unmap(); // 

    printf("App::index DONE \n");
}

void App::sleep(unsigned int microseconds)
{
    printf("App::usleep %d \n", microseconds);
    usleep(microseconds);
}


void App::render()
{
    //printf("App::render\n");
    glUseProgram (m_shader);
    glBindVertexArray (m_vao);
    glDrawArrays (GL_LINE_LOOP, 0, m_nvert);
}

template <typename T>
void App::mapped_dump(unsigned int id)
{
    GLenum target = GL_ARRAY_BUFFER ;
    glBindBuffer( target, id );
    T* data = static_cast<T*>( glMapBuffer(target, GL_READ_ONLY ));
    if(data)
    {
        printf("App::mapped_dump %d \n", id );
        for(unsigned int i=0 ; i < std::min(10u,m_nvert) ; i++)
        {
            unsigned int* d = data + i*4 ; 
            printf(" %3u : %2u %2u %2u %2u \n", i, d[0], d[1], d[2], d[3] ); 
         } 
    }
    else
    {
        printf("App::mapped_dump %d : NULL map \n", id );
    }
    assert( GL_TRUE == glUnmapBuffer( target ));
    glBindBuffer( target, 0 );
}

void App::dump_seq(const char* msg)
{
    if(m_seq)
    {
        printf("App::dump_seq %s \n", msg );
        mapped_dump<unsigned int>(m_seq->id);
    }
}
void App::dump_sel(const char* msg)
{
    if(m_sel)
    {
        printf("App::dump_sel %s \n", msg );
        mapped_dump<unsigned int>(m_sel->id);
    }
}


int main () 
{
    init_glfw();
    init_gl();                                 

    App app(4000) ; 

    app.generate(0.5f);
    //app.tgenerate(1.0f);
    app.dump_sel("after generate");

    app.index();
    //app.sleep(1000000);
    app.dump_sel("after index");


    while (!glfwWindowShouldClose (window)) 
    {
          glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

          app.render();
          app.tscale(1.001f);

          glfwPollEvents ();
          glfwSwapBuffers (window);
    }
    glfwTerminate();

    return 0;
}




