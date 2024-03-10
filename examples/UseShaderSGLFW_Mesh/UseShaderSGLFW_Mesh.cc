/**
examples/UseShaderSGLFW/UseShaderSGLFW_Mesh.cc
================================================

Started from ~/o/examples/UseShaderSGLFW and 
transitioned from single triangle to a mesh.::

    ~/o/examples/UseShaderSGLFW_Mesh/go.sh 
    SHADER=wireframe ~/o/examples/UseShaderSGLFW_Mesh/go.sh 
    SHADER=normal ~/o/examples/UseShaderSGLFW_Mesh/go.sh 

**/

#include "SGLFW.h"
#include "SGLM.h"
#include "SMesh.h"
#include "ssys.h"

int main()
{
    SMesh* mesh = SMesh::Load("$MESH_FOLD"); 
    sframe fr ; fr.ce = make_float4(mesh->ce.x, mesh->ce.y, mesh->ce.z, mesh->ce.w ); 
    SGLM gm ; 
    //gm.setLookRotation( 45.f , {1.f, 1.f, 1.f } );  // angleAxis quaternion 
    gm.set_frame(fr) ; std::cout << gm.desc() ;  // HMM: set_ce ? avoid frame when not needed ?

    SGLFW gl(gm, gm.Width(), gm.Height(), mesh->name ); 
    gl.createProgram("$SHADER_FOLD"); 

    float* MVP_ptr = gm.MVP_ptr ;
    //float* MVP_ptr = gm.IDENTITY_ptr ;
    //float* MVP_ptr = gm.MV_ptr ;

    gl.locateMVP("MVP",  MVP_ptr ); 

    SGLFW_VAO vao ;  // vao: establishes context for OpenGL attrib state and element array (not vbuf,nbuf)
    SGLFW_Buffer ibuf( mesh->tri->arr_bytes(), mesh->tri->cvalues<int>()  , GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW ); 
    SGLFW_Buffer vbuf( mesh->vtx->arr_bytes(), mesh->vtx->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    SGLFW_Buffer nbuf( mesh->nrm->arr_bytes(), mesh->nrm->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 

    gl.enableAttrib( "vPos", "3,GL_FLOAT,GL_FALSE,12,0,false" );  // 3:vec3, 12:byte_stride 0:byte_offet
    gl.enableAttrib( "vNrm", "3,GL_FLOAT,GL_FALSE,12,0,false" ); 

    int num = ssys::getenvint("NUM",mesh->indices_num) ; 
    int off = ssys::getenvint("OFF",mesh->indices_offset) ; 

    while(gl.renderloop_proceed())
    {
        gl.renderloop_head();  // calls gl.updateMVP
        glDrawElements(GL_TRIANGLES, num, GL_UNSIGNED_INT, (GLvoid*)(sizeof(GLuint) * off ));
        gl.renderloop_tail(); 
    }
    exit(EXIT_SUCCESS);
}

