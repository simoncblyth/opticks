/**
UseGeometryShader : flying point visualization of uncompressed step-by-step photon records
=============================================================================================

Usage::

    ~/o/examples/UseGeometryShader/build.sh                          ## non-CMake build
    SHADER=rec_flying_point ~/o/examples/UseGeometryShader/build.sh
    SHADER=pos              ~/o/examples/UseGeometryShader/build.sh

    ~/o/examples/UseGeometryShader/go.sh      ## CMake build

See also::

    ~/o/examples/UseGeometryShader/run.sh 
    

* started from https://www.glfw.org/docs/latest/quick.html#quick_example
* reference on geometry shaders https://open.gl/geometry
* see notes/issues/geometry-shader-flying-photon-visualization.rst

TODO: 

0. avoid hardcoded assumptions on record positions and times

1. more encapsulation/centralization of GLFW/OpenGL mechanics and viz math down 
   into the header only imps: SGLFW.h and SGLM.h

2. WASD navigation controls using SGLFW callback passing messages to SGLM::INSTANCE

3. bring back seqhis photon history selection 

4. DONE: rotation 

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


int main(int argc, char** argv)
{
    const char* RECORD_PATH = "$RECORD_FOLD/record.npy" ; // expect shape like (10000, 10, 4, 4) of type np.float32
    NP* _a = NP::Load(RECORD_PATH) ;   
    NP* a = NP::MakeNarrowIfWide(_a) ; 


    if(a==nullptr) std::cout << "FAILED to load RECORD_PATH [" << RECORD_PATH << "]" << std::endl ;
    if(a==nullptr) std::cout << " CREATE IT WITH [TEST=make_record_array ~/o/sysrap/tests/sphoton_test.sh] " << std::endl ; 
    assert(a); 


    float ADHOC = ssys::getenvfloat("ADHOC", 1.f) ; 
    std::cout << "ADHOC : " << ADHOC << std::endl ;   
    if(ADHOC!=1.f)
    {
        float* aa = a->values<float>(); 
        assert( a->shape.size() == 4 ); 
        int ni = a->shape[0] ; 
        int nj = a->shape[1] ; 
        int nk = a->shape[2] ; 
        int nl = a->shape[3] ; 

        for(int i=0 ; i < ni ; i++)
        for(int j=0 ; j < nj ; j++)
        for(int k=0 ; k < nk ; k++)
        for(int l=0 ; l < nl ; l++)
        {
            int idx = i*nj*nk*nl + j*nk*nl + k*nl + l ;
            if( k == 0 && l < 3 ) aa[idx] = ADHOC*aa[idx] ;  
        }
    }


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

    sfr fr ; 
    fr.set_ce(&ce.x);  

    SGLM gm ; 
    gm.set_frame(fr); 
    gm.dump();

    SGLFW gl(gm);   

    SGLFW_Program prog("$SHADER_FOLD", nullptr, nullptr, nullptr, nullptr, nullptr ); 
    prog.use(); 


    // The strings below are names of uniforms present in rec_flying_point/geom.glsl and pos/vert.glsl 
    GLint Param_location = prog.getUniformLocation("Param"); 

    SGLFW_VAO vao ; 
    vao.bind(); 
 
    SGLFW_Buffer buf(  a->arr_bytes(), a->bytes(), GL_ARRAY_BUFFER, GL_STATIC_DRAW ); 
    buf.bind();
    buf.upload();


    std::string rpos_spec = a->get_meta<std::string>("rpos", "4,GL_FLOAT,GL_FALSE,64,0,false");  
    std::cout 
        << "UseGeometryShader.main "
        << " rpos_spec [" << rpos_spec << "]" 
        << std::endl 
        ;  

    prog.enableVertexAttribArray("rpos", rpos_spec.c_str() ); 

    prog.locateMVP("ModelViewProjection", gm.MVP_ptr ); 


    while (gl.renderloop_proceed())
    {
        gl.renderloop_head(); 

        Param.w += Param.z ;  // input propagation time 
        if( Param.w > Param.y ) Param.w = Param.x ;  // input time : Param.w from .x to .y with .z steps

        //gl.UniformMatrix4fv( gl.mvp_location, mvp );  
        if(Param_location > -1 ) prog.Uniform4fv(      Param_location, glm::value_ptr(Param), false );
        prog.updateMVP();

        GLenum mode = prog.geometry_shader_text ? GL_LINE_STRIP : GL_POINTS ;  
        glDrawArrays(mode, a_first, a_count);


        gl.renderloop_tail();  
    }
    return 0 ; 
}

