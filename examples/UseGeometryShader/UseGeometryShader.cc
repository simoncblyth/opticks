/**
UseGeometryShader : flying point visualization of uncompressed step-by-step photon records
=============================================================================================

* started from https://www.glfw.org/docs/latest/quick.html#quick_example
* reference on geometry shaders https://open.gl/geometry
* see notes/issues/geometry-shader-flying-photon-visualization.rst

TODO: 

1. more encapsulation/centralization of GLFW/OpenGL mechanics and viz math down 
   into the header only imps: SGLFW.hh and SGLM.hh

2. WASD navigation controls using SGLFW callback passing messages to SGLM::INSTANCE

3. expand to working with compressed records when have implemented those

4. bring back seqhis photon history selection 


Both compressed and full record visualization work but currently they 
need adhoc time+space scalings to make them behave similar to each other.
Need a better way to make them behave like each other. 
Attempt to control everything via domain uniforms adds 
lots of complication, is there a better way ?

    ARG=r ./go.sh run   ## this is default 
    ARG=c ./go.sh run

**/

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include "scuda.h"
#include "squad.h"
#include "SGLFW.hh"
#include "SGLM.hh"
#include "NP.hh"

static const char* SHADER_FOLD = getenv("SHADER_FOLD"); 
static const char* vertex_shader_text = NP::ReadString(SHADER_FOLD, "vert.glsl"); 
static const char* geometry_shader_text = NP::ReadString(SHADER_FOLD, "geom.glsl"); 
static const char* fragment_shader_text = NP::ReadString(SHADER_FOLD, "frag.glsl"); 
static const char* ARRAY_FOLD = getenv("ARRAY_FOLD"); 

int main(int argc, char** argv)
{
    const char* ARG = getenv("ARG"); 
    char arg = ARG == nullptr ? 'r' : ARG[0] ; 

    const char* ARRAY_NAME = nullptr ; 
    switch(arg)
    {
        case 'c': ARRAY_NAME = "c.npy" ; break ; 
        case 'r': ARRAY_NAME = "r.npy" ; break ; 
        default : ARRAY_NAME = "r.npy" ; break ; 
    }

    NP* record = NP::Load(ARRAY_FOLD, ARRAY_NAME) ;   // expecting shape like (10000, 10, 4, 4)
    assert(record->shape.size() == 4);   
    bool is_compressed = record->ebyte == 2 ; 
    GLsizei record_count = record->shape[0]*record->shape[1] ;  
    GLint   record_first = 0 ; 

    std::cout 
        << " arg " << arg 
        << " ARRAY_NAME " << ARRAY_NAME  
        << " record " << record->sstr() 
        << " is_compressed " << is_compressed 
        << " record_count " << record_count 
        << std::endl 
        ; 

    float4 post_center = make_float4( 0.f, 0.f, 0.f, 0.f ); 
    float4 post_extent = make_float4( 1.f, 1.f, 1.f, 5.f ); 
    float extent = is_compressed ? 1.f : 1000.f ; 

/*
    if( is_compressed )
    {
        // HMM: where to encapsulate this kinda thing ? cannot be from qudarap : maybe sysrap/squad.h  
        NP* domain = NP::Load(ARRAY_FOLD, "domain.npy") ; 
        if( domain == nullptr ) std::cout << "ERROR : missing domain.npy in ARRAY_FOLD " << ARRAY_FOLD << std::endl ; 
        if( domain == nullptr ) return 0 ;  
        assert( domain->has_shape(2,4,4) && domain->uifc == 'f' && domain->ebyte == 4 ); 
        quad4 dom[2] ; 
        domain->write((float*)&dom[0]);  
        post_center = dom[0].q0.f ; 
        post_extent = dom[0].q1.f ; 
        std::cout << " post_center " << post_center << std::endl ; 
        std::cout << " post_extent " << post_extent << std::endl ; 

        extent = post_extent.x ; 
    }
    else
    {
        //extent = 1000.f ; 
        extent = 1.f ; 
    }
*/

    // Param.w is incremented from .x to .y by ,z  
    glm::vec4 Param(0.f, post_extent.w, post_extent.w/1000.f , 0.f);    // t0, t1, dt, tc 

    // expecting full step record with shape like (10000, 10, 4, 4) and type np.float32 
    // or compressed step record with shape like (10000, 10, 2, 4) and type np.int16 
    //record->set_meta<std::string>("rpos", "4,GL_FLOAT,GL_FALSE,64,0,false" );   
    // ATT CONFIG BELONGS IN QEvent 

    SGLFW sglfw(1280, 720, ARRAY_NAME );    
    sglfw.createProgram(vertex_shader_text, geometry_shader_text, fragment_shader_text ); 

    GLint ModelViewProjection_location = glGetUniformLocation(sglfw.program, "ModelViewProjection");   SGLFW::check(__FILE__, __LINE__);
    GLint Param_location               = glGetUniformLocation(sglfw.program, "Param");                 SGLFW::check(__FILE__, __LINE__);

    GLint post_center_location         = glGetUniformLocation(sglfw.program, "post_center");           SGLFW::check(__FILE__, __LINE__);
    GLint post_extent_location         = glGetUniformLocation(sglfw.program, "post_extent");           SGLFW::check(__FILE__, __LINE__);

    unsigned vao ;                                                                                     SGLFW::check(__FILE__, __LINE__); 
    glGenVertexArrays (1, &vao);                                                                       SGLFW::check(__FILE__, __LINE__);
    glBindVertexArray (vao);                                                                           SGLFW::check(__FILE__, __LINE__);

    GLuint record_buffer ; 
    glGenBuffers(1, &record_buffer);                                                      SGLFW::check(__FILE__, __LINE__);
    glBindBuffer(GL_ARRAY_BUFFER, record_buffer);                                         SGLFW::check(__FILE__, __LINE__);
    glBufferData(GL_ARRAY_BUFFER, record->arr_bytes(), record->bytes(), GL_STATIC_DRAW);  SGLFW::check(__FILE__, __LINE__);

    std::string rpos_spec = record->get_meta<std::string>("rpos", "");  
    sglfw.enableArrayAttribute("rpos", rpos_spec.c_str() ); 
 
    int width, height;
    glfwGetFramebufferSize(sglfw.window, &width, &height); // windows can be resized, so need to grab it 
    std::cout << " width " << width << " height " << height << std::endl ; 

    SGLM sglm ; 
    sglm.width = width ; 
    sglm.height = height ; 
    sglm.zoom = 1.f ;   
    sglm.eye_m.x = -2.f ; 
    sglm.eye_m.y = 0.f ; 
    sglm.eye_m.z = 0.f ; 
    sglm.setExtent(extent); 
    sglm.update(); 
    sglm.dump(); 

    const glm::mat4& world2clip = sglm.world2clip ; 
    const GLfloat* mvp = (const GLfloat*) glm::value_ptr(world2clip) ;  

    bool exitloop(false);
    int renderlooplimit(2000);
    int count(0); 

    while (!glfwWindowShouldClose(sglfw.window) && !exitloop)
    {
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(sglfw.program);

        Param.w += Param.z ; if( Param.w > Param.y ) Param.w = Param.x ; 

        glUniformMatrix4fv(ModelViewProjection_location, 1, GL_FALSE, mvp );
        glUniform4fv(      Param_location,               1, glm::value_ptr(Param) );
        glUniform4fv( post_center_location,              1, &post_center.x );
        glUniform4fv( post_extent_location,              1, &post_extent.x );

        glDrawArrays(GL_LINE_STRIP, record_first, record_count);

        glfwSwapBuffers(sglfw.window);
        glfwPollEvents();
        exitloop = renderlooplimit > 0 && count > renderlooplimit ; 
        count++ ; 
    }
    return 0 ; 
}

