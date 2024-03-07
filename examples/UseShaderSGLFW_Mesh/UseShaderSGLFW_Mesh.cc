/**
examples/UseShaderSGLFW/UseShaderSGLFW_Mesh.cc
================================================

::

    ~/o/examples/UseShaderSGLFW_Mesh/go.sh 

    SHADER=wireframe ~/o/examples/UseShaderSGLFW_Mesh/go.sh 
    SHADER=normal ~/o/examples/UseShaderSGLFW_Mesh/go.sh 

Started from ~/o/examples/UseShaderSGLFW and 
transitioned from single triangle to a mesh. 

**/

#include "NPFold.h"
#include "SGLFW.h"
#include "SGLM.h"
#include "stra.h"
#include "ssys.h"

int main(void)
{
    // HMM: maybe SMesh ? 
    NPFold* fold = NPFold::Load("$MESH_FOLD"); 
    const NP* tri = fold->get("tri"); 
    const NP* _vtx = fold->get("vtx"); 
    NP* _nrm = SGLM::SmoothNormals( _vtx, tri ); // smooth in double precision 
    // narrow to float 
    const NP* vtx = NP::MakeNarrowIfWide(_vtx); 
    const NP* nrm = NP::MakeNarrowIfWide(_nrm); 


    sframe fr ; 
    fr.ce = make_float4( 0.f, 0.f, 0.f,  100.f ); 

    SGLM gm ; 
    gm.set_frame(fr) ; 
    //gm.writeDesc("$FOLD", "A" ); 
    const float* world2clip = (const float*)glm::value_ptr(gm.world2clip) ;
    std::cout << gm.desc() ; 

    assert( tri->shape.size() == 2 ); 
    int indices_count = tri->shape[0]*tri->shape[1] ; 
    GLvoid* indices_offset = (GLvoid*)(sizeof(GLint) * 0) ; 

    std::stringstream ss ; 
    ss << "UseShaderSGLFW_Mesh " 
       << " tri "  << ( tri ? tri->sstr() : "-" ) 
       << " vtx "  << ( vtx ? vtx->sstr() : "-" )
       ;
    std::string str = ss.str();  
    const char* title = str.c_str(); 

    SGLFW gl(gm.Width(), gm.Height(), title ); 
    gl.createProgram("$SHADER_FOLD"); 
    gl.locateMVP("MVP",  world2clip ); 

    SGLFW_Buffer vbuf( vtx->arr_bytes(), vtx->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    SGLFW_Buffer nbuf( nrm->arr_bytes(), nrm->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    
    SGLFW_VAO vao ;  // vao: establishes context for OpenGL attrib state and element array
    SGLFW_Buffer ibuf( tri->arr_bytes(), tri->cvalues<int>()  ,  GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW ); 

    gl.enableAttrib( "vPos", "3,GL_FLOAT,GL_FALSE,12,0,false" );  // 3:vec3, 12:byte_stride
    gl.enableAttrib( "vNrm", "3,GL_FLOAT,GL_FALSE,12,0,false" ); 

    while(gl.renderloop_proceed())
    {
        gl.renderloop_head(); 

        glDrawElements(GL_TRIANGLES, indices_count, GL_UNSIGNED_INT, indices_offset );

        gl.renderloop_tail(); 
    }
    exit(EXIT_SUCCESS);
}
