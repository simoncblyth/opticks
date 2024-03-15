/**
examples/UseShaderSGLFW/UseShaderSGLFW_MeshMesh_Instanced.cc
===============================================================

Started from ~/o/examples/UseShaderSGLFW_MeshMesh and 
adding handling of intance transforms::

    ~/o/examples/UseShaderSGLFW_MeshMesh_Instanced/go.sh 

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


NP* make_instance_transforms()
{
    std::vector<float> vals = {{

       1.f, 0.f, 0.f, 0.f,
       0.f, 1.f, 0.f, 0.f,
       0.f, 0.f, 1.f, 0.f,
       0.f, 0.f, 0.f, 1.f,

       1.f, 0.f, 0.f, 0.f,
       0.f, 1.f, 0.f, 0.f,
       0.f, 0.f, 1.f, 0.f,
       20.f,0.f, 0.f, 1.f 
    }} ; 

    NP* inst = NP::Make<float>( 2, 4, 4) ; 
    inst->read2(vals.data()); 
    return inst ; 
}


int main()
{
    const SMesh* box = SMesh::Load("$MESH_FOLD/Box"); 
    const SMesh* torus = SMesh::Load("$MESH_FOLD/Torus"); 
    NP* inst = make_instance_transforms(); 

    sframe fr ; 
    fr.ce = make_float4(0.f, 0.f, 0.f, 200.f); 

    SGLM gm ; 
    gm.set_frame(fr) ; 
    // TODO: SGLM::set_ce/set_fr ? avoid heavyweight sframe when not needed ?
    std::cout << gm.desc() ;  


    SGLFW gl(gm); 
#ifdef WITH_CUDA_GL_INTEROP 
    SGLFW_CUDA cuda(gm) ; 
#endif


    SGLFW_Program a_prog("$SHADER_FOLD/iwireframe", "vPos", "vNrm", "vInstanceTransform" );
    a_prog.use(); 
    a_prog.locateMVP("MVP",  gm.MVP_ptr );  

    SGLFW_Program b_prog("$SHADER_FOLD/wireframe", "vPos", "vNrm"  );
    b_prog.use(); 
    b_prog.locateMVP("MVP",  gm.MVP_ptr );  

    SGLFW_Render rbox(     box, &a_prog, inst ); 
    SGLFW_Render rtorus( torus, &b_prog ); 

 
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
            rbox.render();
            rtorus.render();
        }
        gl.renderloop_tail();      // swap buffers, poll events
    }
    exit(EXIT_SUCCESS);
}

