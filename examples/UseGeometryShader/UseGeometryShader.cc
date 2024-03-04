/**
UseGeometryShader : flying point visualization of uncompressed step-by-step photon records
=============================================================================================

Usage::

    ~/o/examples/UseGeometryShader/build.sh   ## non-CMake build
    ~/o/examples/UseGeometryShader/go.sh      ## CMake build

* started from https://www.glfw.org/docs/latest/quick.html#quick_example
* reference on geometry shaders https://open.gl/geometry
* see notes/issues/geometry-shader-flying-photon-visualization.rst

TODO: 

0. avoid hardcoded assumptions on record positions and times

1. more encapsulation/centralization of GLFW/OpenGL mechanics and viz math down 
   into the header only imps: SGLFW.h and SGLM.h

2. WASD navigation controls using SGLFW callback passing messages to SGLM::INSTANCE

3. bring back seqhis photon history selection 

4. rotation 


Attempt to control everything via domain uniforms adds lots of complication, 
is there a better way ? 


WIP: remove record array position/time hardcoding
---------------------------------------------------

::

    In [15]: np.min(t.record[:,:,0].reshape(-1,4),axis=0)
    Out[15]: array([-9.,  0., -9.,  0.], dtype=float32)

    In [16]: np.max(t.record[:,:,0].reshape(-1,4),axis=0)
    Out[16]: array([9., 0., 9., 9.], dtype=float32)


**/

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include "ssys.h"
#include "scuda.h"
#include "squad.h"
#include "sframe.h"
#include "SGLFW.h"
#include "SGLM.h"

#include "NP.hh"

static const char* vertex_shader_text = U::ReadString("$SHADER_FOLD/vert.glsl"); 
static const char* geometry_shader_text = U::ReadString("$SHADER_FOLD/geom.glsl"); 
static const char* fragment_shader_text = U::ReadString("$SHADER_FOLD/frag.glsl"); 

int main(int argc, char** argv)
{
    std::cout << " vertex_shader_text " << ( vertex_shader_text ? "YES" : "NO" ) << std::endl ; 
    std::cout << " geometry_shader_text " << ( geometry_shader_text ? "YES" : "NO" ) << std::endl ; 
    std::cout << " fragment_shader_text " << ( fragment_shader_text ? "YES" : "NO" ) << std::endl ; 


    //const char* RECORD_PATH = "$RECORD_FOLD/rec.npy" ; // expect shape like (10000, 10, 2, 4) of type np.int16 [NO LONGER USED]
    const char* RECORD_PATH = "$RECORD_FOLD/record.npy" ; // expect shape like (10000, 10, 4, 4) of type np.float32
    NP* a = NP::Load(RECORD_PATH) ;   
    if(a==nullptr) std::cout << "FAILED to load RECORD_PATH [" << RECORD_PATH << "]" << std::endl ; 
    assert(a); 

    assert(a->shape.size() == 4);   
    bool is_compressed = a->ebyte == 2 ; 
    assert( is_compressed == false ); 

    GLint   a_first = 0 ; 
    GLsizei a_count = a->shape[0]*a->shape[1] ;   // all step points across all photon

    std::cout 
        << "UseGeometryShader.main "
        << " RECORD_PATH " << RECORD_PATH
        << " a.sstr " << a->sstr() 
        << " is_compressed " << is_compressed 
        << " a_first " << a_first 
        << " a_count " << a_count 
        << std::endl 
        ; 

    float4 mn = {} ; 
    float4 mx = {} ; 
    static const int N = 4 ;   

    int item_stride = 4 ; 
    int item_offset = 0 ; 

    a->minmax2D_reshaped<N,float>(&mn.x, &mx.x, item_stride, item_offset ); 
    // temporarily 2D array with item: 4-element space-time points

    // HMM: with sparse "post" cloud this might miss the action
    // by trying to see everything ? 

    float4 ce = scuda::center_extent( mn, mx ); 

    std::cout 
        << "UseGeometryShader.main "
        << std::endl   
        << std::setw(20) << " mn " << mn 
        << std::endl   
        << std::setw(20) << " mx " << mx 
        << std::endl   
        << std::setw(20) << " ce " << ce
        << std::endl   
        ;

    // Param holds time range and step with Param.w incremented in 
    // the render loop to scrub the time from .x to .y by steps of .z  

    float t0 = ssys::getenvfloat("T0", mn.w ); 
    float t1 = ssys::getenvfloat("T1", mx.w ); 
    float ts = ssys::getenvfloat("TS", 1000. ); 

    std::cout 
        << "UseGeometryShader.main "
        << std::endl   
        << " T0 "
        << std::fixed << std::setw(10) << std::setprecision(4) << t0 
        << " T1 "
        << std::fixed << std::setw(10) << std::setprecision(4) << t1
        << " TS "
        << std::fixed << std::setw(10) << std::setprecision(4) << ts
        << std::endl 
        ;
  

    glm::vec4 Param(t0, t1, (t1-t0)/ts, t0);    // t0, t1, dt, tc 

    sframe fr ; 
    fr.ce = ce ; 

    /*
    float4 post_center = make_float4( ce.x, ce.y, ce.z,  mn.w  );        // minimum time as "center" of time 
    float4 post_extent = make_float4( ce.w, ce.w, ce.w,  mx.w - mn.w );  // time range as "extent" of time 

    HMM: THESE CURRENTLY NOT USED IN geom.glsl 
    SHOULD THEY BE ? NO, THE VALUES COME IN THE RECORD POINTS 
    SO THEY DONT NEED TO BE USED THERE.
    AHHA: I RECALL THE ORIGIN OF THESE WAS WHEN WAS USING DOMAIN COMPRESSED REC.
    THE POST_CENTER AND POST_EXTENT WAS USED TO DO THE UNCOMPRESSION.
    THE DETAILS ARE AN ARTIFACT OF HOW THE DOMAIN COMPRESSION WAS DONE.
   
    BUT: THE CENTER EXTENT IS NEEDED FOR sframe IN ORDER TO TARGET THE 
    VIEW AT THE POINTS. 
    */


    SGLM sglm ; 
    sglm.set_frame(fr); 
    sglm.update(); 
    //sglm.dump();

    const char* title = RECORD_PATH ; 
    SGLFW sglfw(sglm.Width(), sglm.Height(), title );   
    sglfw.createProgram(vertex_shader_text, geometry_shader_text, fragment_shader_text ); 

    // The strings below are names of uniforms present in rec_flying_point/geom.glsl
    // SGLFW could hold a map of uniform locations keyed on the names
    // which are grabbed from the shader source by pattern matching uniform lines  

    GLint ModelViewProjection_location = glGetUniformLocation(sglfw.program, "ModelViewProjection");   SGLFW::check(__FILE__, __LINE__);
    GLint Param_location               = glGetUniformLocation(sglfw.program, "Param");                 SGLFW::check(__FILE__, __LINE__);
    //GLint post_center_location         = glGetUniformLocation(sglfw.program, "post_center");           SGLFW::check(__FILE__, __LINE__);
    //GLint post_extent_location         = glGetUniformLocation(sglfw.program, "post_extent");           SGLFW::check(__FILE__, __LINE__);

    unsigned vao ;                  SGLFW::check(__FILE__, __LINE__); 
    glGenVertexArrays (1, &vao);    SGLFW::check(__FILE__, __LINE__);
    glBindVertexArray (vao);        SGLFW::check(__FILE__, __LINE__);

    GLuint a_buffer ; 
    glGenBuffers(1, &a_buffer);                                                 SGLFW::check(__FILE__, __LINE__);
    glBindBuffer(GL_ARRAY_BUFFER, a_buffer);                                    SGLFW::check(__FILE__, __LINE__);
    glBufferData(GL_ARRAY_BUFFER, a->arr_bytes(), a->bytes(), GL_STATIC_DRAW);  SGLFW::check(__FILE__, __LINE__);

    std::string rpos_spec = a->get_meta<std::string>("rpos", "");  
    std::cout 
        << "UseGeometryShader.main "
        << " rpos_spec [" << rpos_spec << "]" 
        << std::endl 
        ;  

    sglfw.enableArrayAttribute("rpos", rpos_spec.c_str() ); 
 
    int width, height;
    glfwGetFramebufferSize(sglfw.window, &width, &height); 
    std::cout 
        << "UseGeometryShader.main "
        << " width " << width << " height " << height 
        << std::endl 
        ; 
    // windows can be resized, so need to grab it 
    // Q: what about resizes during render loop ? HMM: need callback for that ?

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

        Param.w += Param.z ;  // input propagation time 
        if( Param.w > Param.y ) Param.w = Param.x ;  // input time : Param.w from .x to .y with .z steps

        glUniformMatrix4fv(ModelViewProjection_location, 1, GL_FALSE, mvp );
        glUniform4fv(      Param_location,               1, glm::value_ptr(Param) );
        //glUniform4fv( post_center_location,              1, &post_center.x );
        //glUniform4fv( post_extent_location,              1, &post_extent.x );

        GLenum prim = geometry_shader_text ? GL_LINE_STRIP : GL_POINTS ;  
        glDrawArrays(prim, a_first, a_count);

        glfwSwapBuffers(sglfw.window);
        glfwPollEvents();

        exitloop = renderlooplimit > 0 && count > renderlooplimit ; 
        count++ ; 
    }
    return 0 ; 
}

