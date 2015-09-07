#include <stdio.h>
#include "assert.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "OBuffer.hh"
#include "TAry.hh"



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

template <typename T>
GLuint init_vbo(unsigned int nvert, unsigned int nelem, void* data )
{
    GLuint vbo ; 
    glGenBuffers (1, &vbo);
    glBindBuffer (GL_ARRAY_BUFFER, vbo);

    //GLenum usage = GL_STATIC_DRAW ; 
    GLenum usage = GL_DYNAMIC_DRAW ; 
    //GLenum usage = GL_STREAM_DRAW ; 
    switch(usage)
    {
      case GL_STATIC_DRAW: printf("STATIC_DRAW\n");break;
      case GL_DYNAMIC_DRAW: printf("DYNAMIC_DRAW\n");break;
      case GL_STREAM_DRAW:  printf("STREAM_DRAW\n");break;
    }
    
    glBufferData (GL_ARRAY_BUFFER, nvert * nelem * sizeof (T), data, usage );
    return vbo ; 
}



class App {
    public:
       static const char* CMAKE_TARGET ; 
       static const char* vertex_shader ; 
       static const char* fragment_shader ; 
    public:
       enum { e_vtx, e_sel } ;
       enum { raygen_minimal_entry, raygen_dump_entry, num_entry } ;
    public:
       App(unsigned int nvert);
    private:
       void init();
       void init_vao();
       void init_shader();
       void init_glbo();
       void init_optix();
       static float* make_triangle_data(float s);
       void addRayGenerationProgram( const char* ptxname, const char* progname, unsigned int entry );
    public:
       void generate(float radius);
       void index();
       void render();
    private:
       unsigned int m_device ;  
    private:
       // OpenGL names
       unsigned int  m_vao ;  
       unsigned int  m_vtx ;  
       float*        m_vtx_data ; 
       unsigned int  m_sel ;  
       unsigned int* m_sel_data ; 
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
   m_vtx(0),
   m_vtx_data(NULL),
   m_sel(0),
   m_sel_data(NULL),
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
    init_glbo();
    init_vao();
    init_shader();
    init_optix();  
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


void App::init_glbo()
{
    if(m_nvert == 3) m_vtx_data = make_triangle_data(0.9f);
    m_vtx = init_vbo<float>(m_nvert, 4, m_vtx_data );

    m_sel_data = new unsigned int[m_nvert*4] ;
    for(unsigned int i=0 ; i < m_nvert ; i++)
    {
        m_sel_data[i*4+0] = i % 4 ; 
        m_sel_data[i*4+1] = 1u ; 
        m_sel_data[i*4+2] = 1u ; 
        m_sel_data[i*4+3] = 1u ; 
    }
    m_sel = init_vbo<unsigned int>(m_nvert, 4, m_sel_data );

    printf("App::init_glbo m_vtx %d m_sel %d \n", m_vtx, m_sel );
}


void App::init_vao()
{
    glGenVertexArrays (1, &m_vao);
    glBindVertexArray (m_vao);
    printf("App::init_vao m_vao %d m_vtx %d m_sel %d \n", m_vao, m_vtx, m_sel);

    glBindBuffer (GL_ARRAY_BUFFER, m_vtx);
    glVertexAttribPointer (e_vtx, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray (e_vtx);

    glBindBuffer (GL_ARRAY_BUFFER, m_sel);
    glVertexAttribIPointer (e_sel, 4, GL_UNSIGNED_INT, 0, NULL);  // glVertexAttrib*I*Pointer 
    glEnableVertexAttribArray (e_sel);
}

/*

//"  colour = vsel.x == 0 ? white : red ;"

"  colour = green ; "
"  vec4 white = vec4 (1.0, 1.0, 1.0, 1.0);"
"  vec4 red   = vec4 (1.0, 0.0, 0.0, 1.0);"
"  vec4 green = vec4 (0.0, 1.0, 0.0, 1.0);"
"  vec4 blue  = vec4 (0.0, 0.0, 1.0, 1.0);"


*/


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

    addRayGenerationProgram("circle.cu.ptx", "circle_make_vertices", raygen_minimal_entry );
    addRayGenerationProgram("circle.cu.ptx", "circle_dump",          raygen_dump_entry );

    m_vtxbuf = new OBuffer(m_context, m_vtx, "vtx_buffer", m_nvert, OBuffer::RW ); 
    m_selbuf = new OBuffer(m_context, m_sel, "sel_buffer", m_nvert, OBuffer::W ); 

}

void App::generate(float radius)
{
    if(m_nvert == 3) return ;

    printf("App::generate %10.4f \n", radius);

    m_vtxbuf->mapGLToOptiX();  // createBufferFromGLBO

    m_context->validate();
    m_context->compile();

    m_context["radius"]->setFloat(radius) ; 
    m_context->launch(raygen_minimal_entry, m_nvert, 1); // generate vertices

    m_vtxbuf->unmapGLToOptiX();
}


void App::index()
{
    if(m_nvert == 3) return ;

    printf("App::index \n");

    m_selbuf->mapGLToCUDA();  

    TAry ta(m_selbuf->getRawPointer(), m_selbuf->getSize());
    ta.transform();

    m_selbuf->unmapGLToCUDA();

    m_selbuf->streamSync();

    printf("App::index DONE \n");
}





/*
void App::russian_doll(float radius, float factor)
{
    // output suggests thrust is modifying OK, but OpenGL doesnt see it 
    printf("GLOptiXThrust::russian_doll\n");
    m_vtxbuf->mapGLToOptiX();  // createBufferFromGLBO
    m_context->validate();
    m_context->compile();
    m_context["radius"]->setFloat(radius) ; 
    m_context->launch(raygen_minimal_entry, m_nvert, 1); // generate vertices
    {
        m_vtxbuf->mapOptiXToCUDA();  // getDevicePointer
        thrust_transform(factor);
        m_vtxbuf->unmapOptiXToCUDA();
    }
    m_vtxbuf->unmapGLToOptiX();
    printf("App::russian_doll DONE\n");
}
*/

void App::render()
{
    printf("App::render\n");

    glUseProgram (m_shader);
    glBindBuffer (GL_ARRAY_BUFFER, m_vtx);  // is this needed, or does the vao encompass this 
    glBindVertexArray (m_vao);
    glDrawArrays (GL_LINE_LOOP, 0, m_nvert);
}



int main () 
{
    init_glfw();
    init_gl();                                 

    App app(300) ; 

    app.generate(0.5f);
    app.index();

    glFinish(); 

    while (!glfwWindowShouldClose (window)) 
    {
          glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

          app.render();

          glfwPollEvents ();
          glfwSwapBuffers (window);
    }
    glfwTerminate();

    return 0;
}




