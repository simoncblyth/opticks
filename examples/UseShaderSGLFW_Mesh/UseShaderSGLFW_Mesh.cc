/**
examples/UseShaderSGLFW/UseShaderSGLFW_Mesh.cc
================================================

::

    ~/o/examples/UseShaderSGLFW_Mesh/go.sh 

    SHADER=wireframe ~/o/examples/UseShaderSGLFW_Mesh/go.sh 
    SHADER=normal ~/o/examples/UseShaderSGLFW_Mesh/go.sh 

Started from ~/o/examples/UseShaderSGLFW and 
transitioned from single triangle to a mesh. 

HMM : how coupled do SGLFW and SGLM need to be ? 
--------------------------------------------------

HMM: can get away with just the float* pointer for finding the matrix
but need to call SGLM::update after changing view param
so does SGLFW needs to hold the SGLM ? Or can a std::function 
argument be used to keep the two decoupled at arms length ? 

At first glance it look like need to tightly couple, 
as the key callbacks(gleq events) need to drive 
the SGLM interface and cause the matrix and 
potentially other uniforms to be updated. 

But could avoid that by adding a text interface to SGLM, 
that could be used over UDP in future. 
This means the keys just result in sending text commands
via the std::function<int(std::string)> 

**/

#include "NPFold.h"
#include "SGLFW.h"
#include "SGLM.h"
#include "stra.h"
#include "ssys.h"

int main()
{
    NPFold* fold = NPFold::Load("$MESH_FOLD");   // HMM: maybe SMesh to encapsulate this
    const NP* tri = fold->get("tri"); 
    const NP* _vtx = fold->get("vtx"); 
    NP* _nrm = SGLM::SmoothNormals( _vtx, tri ); // smooth in double precision 
    // narrow to float 
    const NP* vtx = NP::MakeNarrowIfWide(_vtx); 
    const NP* nrm = NP::MakeNarrowIfWide(_nrm); 

    sframe fr ; 
    fr.ce = make_float4( 0.f, 0.f, 0.f,  100.f );  // could extract CE from the bbox of vertices  

    SGLM gm ; 
    gm.set_frame(fr) ; 
    std::cout << gm.desc() ; 

    //gm.writeDesc("$FOLD", "A" ); 
    const float* world2clip = (const float*)glm::value_ptr(gm.world2clip) ;

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

    SGLFW gl(gm.Width(), gm.Height(), title, (SCMD*)&gm ); 
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
        gl.renderloop_head();  // calls gl.updateMVP

        glDrawElements(GL_TRIANGLES, indices_count, GL_UNSIGNED_INT, indices_offset );

        gl.renderloop_tail(); 
    }
    exit(EXIT_SUCCESS);
}
