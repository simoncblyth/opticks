/**
examples/UseShaderSGLFW/UseShaderSGLFW_MeshMesh.cc
===================================================

Started from ~/o/examples/UseShaderSGLFW_Mesh and 
transitioned to multiple meshes::

    ~/o/examples/UseShaderSGLFW_MeshMesh/go.sh 

See also::

    ~/o/u4/tests/U4Mesh_test.sh 

    SOLID=Cons ~/o/u4/tests/U4Mesh_test.sh ana
    SOLID=Tubs ~/o/u4/tests/U4Mesh_test.sh ana

    ~/o/sysrap/tests/SMesh_test.sh 
    SOLID=Tet ~/o/sysrap/tests/SMesh_test.sh run
    SOLID=Orb ~/o/sysrap/tests/SMesh_test.sh build_run

**/

#include "ssys.h"
#include "SMesh.h"
#include "SGLM.h"
#include "SGLFW.h"

int main()
{
    const SMesh* box = SMesh::Load("$MESH_FOLD/Box"); 
    const SMesh* torus = SMesh::Load("$MESH_FOLD/Torus"); 
    //const SMesh* other = SMesh::Load("$STREE_MESH_FOLD/$STREE_MESH_SOLID"); 
    const SMesh* other = SMesh::Load("$SCENE_FOLD/scene/mesh_grup/1"); 


    sfr fr ; 
    fr.set_extent(100.);  

    SGLM gm ; 
    gm.set_frame(fr) ; 
    // TODO: SGLM::set_ce/set_fr ? avoid heavyweight sframe when not needed ?
    std::cout << gm.desc() ;  


    SGLFW gl(gm); 
#ifdef WITH_CUDA_GL_INTEROP 
    SGLFW_CUDA cuda(gm) ; 
#endif

    SGLFW_Program prog("$SHADER_FOLD", "vPos", "vNrm", nullptr, "MVP", gm.MVP_ptr );
    SGLFW_Mesh _box(    box  ); 
    SGLFW_Mesh _torus( torus ); 
    SGLFW_Mesh _other( other ); 
 
    while(gl.renderloop_proceed())
    {
        gl.renderloop_head();  // clears 
        if( gl.toggle.cuda )
        {
#ifdef WITH_CUDA_GL_INTEROP 
            cuda.fillOutputBuffer(); 
            cuda.displayOutputBuffer(gl.window);
#endif
        }
        else
        {
            _box.render(&prog);
            _torus.render(&prog);
            _other.render(&prog);
        }
        gl.renderloop_tail();      // swap buffers, poll events
    }
    exit(EXIT_SUCCESS);
}

