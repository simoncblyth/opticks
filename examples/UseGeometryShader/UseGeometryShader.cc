/**
UseGeometryShader : flying point visualization of uncompressed step-by-step photon records
=============================================================================================

* started from https://www.glfw.org/docs/latest/quick.html#quick_example
* reference on geometry shaders https://open.gl/geometry
* see notes/issues/geometry-shader-flying-photon-visualization.rst

TODO: 

1. more encapsulation/centralization of GLFW/OpenGL mechanics and viz math down 
   into the header only imps: SGLFW.h and SGLM.h

2. WASD navigation controls using SGLFW callback passing messages to SGLM::INSTANCE

3. bring back seqhis photon history selection 

4. rotation 


Both compressed and full record visualization work but currently they 
need adhoc time+space scalings to make them behave similar to each other.
Need a better way to make them behave like each other and automate the setting 
of extents. 

Attempt to control everything via domain uniforms adds lots of complication, 
is there a better way ?

    ARG=r ./go.sh run   ## this is default 
    ARG=c ./go.sh run

**/

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

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
    //const char* ARRAY_PATH = "$ARRAY_FOLD/rec.npy" ; // expect shape like (10000, 10, 2, 4) of type np.int16 [NO LONGER USED]
    const char* ARRAY_PATH = "$ARRAY_FOLD/record.npy" ; // expect shape like (10000, 10, 4, 4) of type np.float32
    NP* a = NP::Load(ARRAY_PATH) ;   
    if(a==nullptr) std::cout << "FAILED to load [" << ARRAY_PATH << "]" << std::endl ; 
    assert(a); 

    assert(a->shape.size() == 4);   
    bool is_compressed = a->ebyte == 2 ; 
    assert( is_compressed == false ); 

    GLint   a_first = 0 ; 
    GLsizei a_count = a->shape[0]*a->shape[1] ;   // all step points across all photon

    std::cout 
        << " ARRAY_PATH " << ARRAY_PATH
        << " a.sstr " << a->sstr() 
        << " is_compressed " << is_compressed 
        << " a_first " << a_first 
        << " a_count " << a_count 
        << std::endl 
        ; 

    float4 post_center = make_float4( 0.f, 0.f, 0.f,  0.f ); 
    float4 post_extent = make_float4( 1.f, 1.f, 1.f, 10.f ); 

    // Param.w is incremented from .x to .y by ,z  
    glm::vec4 Param(0.f, post_extent.w, post_extent.w/1000.f , 0.f);    // t0, t1, dt, tc 

    sframe fr ; 
    fr.ce.x = 0.f ; 
    fr.ce.y = 0.f ; 
    fr.ce.z = 0.f ; 
    fr.ce.w = 10.f ; 

    SGLM sglm ; 
    sglm.set_frame(fr); 
    sglm.update(); 
    //sglm.dump();

    const char* title = ARRAY_PATH ; 
    SGLFW sglfw(sglm.Width(), sglm.Height(), title );   
    sglfw.createProgram(vertex_shader_text, geometry_shader_text, fragment_shader_text ); 

    // The four strings below are names present in rec_flying_point/geom.glsl
    // SGLFW could hold a map of uniform locations keyed on the names
    // which are grabbed from the shader source by pattern matching uniform lines  

    GLint ModelViewProjection_location = glGetUniformLocation(sglfw.program, "ModelViewProjection");   SGLFW::check(__FILE__, __LINE__);
    GLint Param_location               = glGetUniformLocation(sglfw.program, "Param");                 SGLFW::check(__FILE__, __LINE__);
    GLint post_center_location         = glGetUniformLocation(sglfw.program, "post_center");           SGLFW::check(__FILE__, __LINE__);
    GLint post_extent_location         = glGetUniformLocation(sglfw.program, "post_extent");           SGLFW::check(__FILE__, __LINE__);

    unsigned vao ;                  SGLFW::check(__FILE__, __LINE__); 
    glGenVertexArrays (1, &vao);    SGLFW::check(__FILE__, __LINE__);
    glBindVertexArray (vao);        SGLFW::check(__FILE__, __LINE__);

    GLuint a_buffer ; 
    glGenBuffers(1, &a_buffer);                                                 SGLFW::check(__FILE__, __LINE__);
    glBindBuffer(GL_ARRAY_BUFFER, a_buffer);                                    SGLFW::check(__FILE__, __LINE__);
    glBufferData(GL_ARRAY_BUFFER, a->arr_bytes(), a->bytes(), GL_STATIC_DRAW);  SGLFW::check(__FILE__, __LINE__);

    std::string rpos_spec = a->get_meta<std::string>("rpos", "");  
    std::cout << " rpos_spec [" << rpos_spec << "]" << std::endl ;  
    sglfw.enableArrayAttribute("rpos", rpos_spec.c_str() ); 
 
    int width, height;
    glfwGetFramebufferSize(sglfw.window, &width, &height); // windows can be resized, so need to grab it 
    std::cout << " width " << width << " height " << height << std::endl ; 

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

        Param.w += Param.z ; if( Param.w > Param.y ) Param.w = Param.x ;  // cycling time : Param.w from .x to .y with .z steps

        glUniformMatrix4fv(ModelViewProjection_location, 1, GL_FALSE, mvp );
        glUniform4fv(      Param_location,               1, glm::value_ptr(Param) );
        glUniform4fv( post_center_location,              1, &post_center.x );
        glUniform4fv( post_extent_location,              1, &post_extent.x );

        glDrawArrays(GL_LINE_STRIP, a_first, a_count);

        glfwSwapBuffers(sglfw.window);
        glfwPollEvents();

        exitloop = renderlooplimit > 0 && count > renderlooplimit ; 
        count++ ; 
    }
    return 0 ; 
}

