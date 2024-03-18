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

#include "ssys.h"
#include "SMesh.h"
#include "SGLM.h"
#include "SGLFW.h"

int main()
{
    SMesh* mesh = SMesh::Load("$MESH_FOLD"); 

    sframe fr ; fr.ce = make_float4(mesh->ce.x, mesh->ce.y, mesh->ce.z, mesh->ce.w ); 
    SGLM gm ; 
    //gm.setLookRotation( 45.f , {1.f, 1.f, 1.f } );  // angleAxis quaternion 
    gm.set_frame(fr) ; std::cout << gm.desc() ;  // HMM: set_ce ? avoid frame when not needed ?

    SGLFW gl(gm, mesh->name ); 

#ifdef WITH_CUDA_GL_INTEROP 
    SGLFW_CUDA cuda(gm) ; 
#endif

    SGLFW_Program prog("$SHADER_FOLD", "vPos", "vNrm" );
    prog.use(); 
    prog.locateMVP("MVP",  gm.MVP_ptr );  


    SGLFW_VAO vao ;  // vao: establishes context for OpenGL attrib state and element array (not vbuf,nbuf)
    vao.bind(); 

    SGLFW_Buffer ibuf( mesh->tri->arr_bytes(), mesh->tri->cvalues<int>()  , GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW ); 
    ibuf.bind();
    ibuf.upload(); 

    SGLFW_Buffer vbuf( mesh->vtx->arr_bytes(), mesh->vtx->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    vbuf.bind();
    vbuf.upload(); 

    SGLFW_Buffer nbuf( mesh->nrm->arr_bytes(), mesh->nrm->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    nbuf.bind();
    nbuf.upload(); 


    while(gl.renderloop_proceed())
    {
        if( gl.toggle.cuda )
        {
#ifdef WITH_CUDA_GL_INTEROP 
             cuda.fillOutputBuffer(); 
             cuda.displayOutputBuffer(gl.window);
#endif
        }
        else
        {
             gl.renderloop_head();  // clears 
             prog.use(); 
             vao.bind(); 

             vbuf.bind();
             prog.enableVertexAttribArray( prog.vtx_attname, SMesh::VTX_SPEC ); 

             nbuf.bind();
             prog.enableVertexAttribArray( prog.nrm_attname, SMesh::NRM_SPEC ); 

             // NB: careful with the ordering of the above or the OpenGL state machine will bite you : 
             // the vPos and vNrm attribs needs to ne enabled after the appropriate buffer is made THE active GL_ARRAY_BUFFER

             ibuf.bind();

             prog.updateMVP();

             glDrawElements(GL_TRIANGLES, mesh->indices_num(), GL_UNSIGNED_INT, (GLvoid*)(sizeof(GLuint) * mesh->indices_offset() ));
             // HMM: prog.draw ?
        }
        gl.renderloop_tail();          // swap buffers, poll events
    }
    exit(EXIT_SUCCESS);
}


