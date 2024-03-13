/**
examples/UseShaderSGLFW/UseShaderSGLFW_Mesh.cc
================================================

Started from ~/o/examples/UseShaderSGLFW and 
transitioned from single triangle to a mesh.::

    ~/o/examples/UseShaderSGLFW_Mesh/go.sh 
    SHADER=wireframe ~/o/examples/UseShaderSGLFW_Mesh/go.sh 
    SHADER=normal ~/o/examples/UseShaderSGLFW_Mesh/go.sh 

See also::

    ~/o/u4/tests/U4Mesh_test.sh 

    SOLID=Cons ~/o/u4/tests/U4Mesh_test.sh ana
    SOLID=Tubs ~/o/u4/tests/U4Mesh_test.sh ana

    ~/o/sysrap/tests/SMesh_test.sh 
    SOLID=Tet ~/o/sysrap/tests/SMesh_test.sh run

**/

#include "SGLFW.h"
#include "SGLM.h"
#include "SMesh.h"
#include "ssys.h"

int main()
{
    SMesh* mesh = SMesh::Load("$MESH_FOLD"); 
    int num = ssys::getenvint("NUM",mesh->indices_num) ; 
    int off = ssys::getenvint("OFF",mesh->indices_offset) ; 

    sframe fr ; fr.ce = make_float4(mesh->ce.x, mesh->ce.y, mesh->ce.z, mesh->ce.w ); 
    SGLM gm ; 
    //gm.setLookRotation( 45.f , {1.f, 1.f, 1.f } );  // angleAxis quaternion 
    gm.set_frame(fr) ; std::cout << gm.desc() ;  // HMM: set_ce ? avoid frame when not needed ?

    SGLFW gl(gm, mesh->name ); 


    // HMM: need separate object for the program and the gl 
    // to easily not use the program 

    gl.createProgram("$SHADER_FOLD"); 
    gl.useProgram(); 
    gl.locateMVP("MVP",  gm.MVP_ptr );  // <-- 


    SGLFW_VAO vao ;  // vao: establishes context for OpenGL attrib state and element array (not vbuf,nbuf)
    vao.bind(); 

    SGLFW_Buffer ibuf( mesh->tri->arr_bytes(), mesh->tri->cvalues<int>()  , GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW ); 
    ibuf.bind();
    ibuf.upload(); 

    SGLFW_Buffer vbuf( mesh->vtx->arr_bytes(), mesh->vtx->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    vbuf.bind();
    vbuf.upload(); 
    gl.enableAttrib( "vPos", "3,GL_FLOAT,GL_FALSE,12,0,false" );  // 3:vec3, 12:byte_stride 0:byte_offet

    SGLFW_Buffer nbuf( mesh->nrm->arr_bytes(), mesh->nrm->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    nbuf.bind();
    nbuf.upload(); 
    gl.enableAttrib( "vNrm", "3,GL_FLOAT,GL_FALSE,12,0,false" ); 

    // NB: careful with the ordering of the above or the OpenGL state machine will bite you : 
    // the vPos and vNrm attribs needs to ne enabled after the appropriate buffer is made THE active GL_ARRAY_BUFFER

 
    while(gl.renderloop_proceed())
    {
        if( gl.toggle.cuda )
        {
             gl.fillOutputBuffer(); 
             gl.displayOutputBuffer(); 
        }
        else
        {
             gl.renderloop_head();  // clears 
             gl.useProgram(); 
             vao.bind(); 

             glDrawElements(GL_TRIANGLES, num, GL_UNSIGNED_INT, (GLvoid*)(sizeof(GLuint) * off ));
        }

        gl.renderloop_update_state(); // listens for inputs calls gl.updateMVP
        gl.renderloop_tail();        // swap buffers, poll events
    }
    exit(EXIT_SUCCESS);
}


/**

 HMM : CAN DO ONE OR THE OTHER : CANNOT FLIP BETWEEN 

**/

