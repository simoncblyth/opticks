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


Issue : Linux instanced not showing up, Darwin OK
-------------------------------------------------- 

The render just does not appear for those solids

* could be down to the dirtry JUNO env (eg GLEW from ROOT)

  * NOPE : trying with cleaner .opticks_config env 
    with glfw and glew installed libs makes no difference



**/

#include "ssys.h"
#include "SMesh.h"
#include "SGLM.h"
#include "SGLFW.h"


NP* make_instance_transforms()
{
    std::vector<float> vals = {{

       1.f,    0.f,    0.f,   0.f,
       0.f,    1.f,    0.f,   0.f,
       0.f,    0.f,    1.f,   0.f,
       0.f,    0.f, -100.f,   1.f,

       1.f,     0.f,   0.f,   0.f,
       0.f,     1.f,   0.f,   0.f,
       0.f,     0.f,   1.f,   0.f,
       0.f,     0.f,   0.f,   1.f,

       1.f,     0.f,   0.f,   0.f,
       0.f,     1.f,   0.f,   0.f,
       0.f,     0.f,   1.f,   0.f,
       0.f,     0.f, 100.f,   1.f,
 
    }} ; 

    NP* inst = NP::Make<float>( 3, 4, 4) ; 
    inst->read2(vals.data()); 
    return inst ; 
}


int main()
{
    std::vector<std::string> solid = {{"Box", "Torus", "Cons", "Tet", "Orb", "Tubs" }} ; 
    int num_solid = solid.size(); 

    std::vector<const SMesh*> mesh ; 
    for(int i=0 ; i < num_solid ; i++)
    {
        glm::tmat4x4<double> tr = stra<double>::Translate( (i - num_solid/2)*100. , 0., 0. , 1. );  
        const SMesh* _mesh = SMesh::Load("$MESH_FOLD", solid[i].c_str(), &tr) ; 
        mesh.push_back(_mesh); 
    } 

    //int INST = ssys::getenvint("INST",0) ; 
    //NP* inst = INST ? make_instance_transforms() : nullptr ; 
    NP* inst = make_instance_transforms() ; 

    std::cout 
        << " inst " << ( inst ? inst->sstr() : "-" ) 
        << std::endl 
        ;

    sframe fr ; 
    fr.ce = make_float4(0.f, 0.f, 0.f, 1000.f); 

    SGLM gm ; 
    gm.set_frame(fr) ; 
    // TODO: SGLM::set_ce/set_fr ? avoid heavyweight sframe when not needed ?
    //std::cout << gm.desc() ;  


    SGLFW gl(gm); 
#ifdef WITH_CUDA_GL_INTEROP 
    SGLFW_CUDA cuda(gm) ; 
#endif


    SGLFW_Program prog("$SHADER_FOLD/wireframe", "vPos", "vNrm" ); 
    prog.use(); 
    prog.locateMVP("MVP",  gm.MVP_ptr );  

    SGLFW_Program iprog("$SHADER_FOLD/iwireframe", "vPos", "vNrm", "vInstanceTransform" ); 
    iprog.use(); 
    iprog.locateMVP("MVP", gm.MVP_ptr );  
 

    std::vector<SGLFW_Render*> render ; 
    int num_mesh = mesh.size(); 
    for(int i=0 ; i < num_mesh ; i++)
    {
        const SMesh* _mesh = mesh[i] ; 

        bool do_instanced = i % 2 == 0  ; 

        SGLFW_Program* _prog = do_instanced ? &iprog : &prog ;  
        NP* _inst = do_instanced ? inst : nullptr ; 

        SGLFW_Render* _render = new SGLFW_Render(_mesh, _prog, _inst );  
        render.push_back(_render); 
    }
    int num_render = render.size(); 


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
            for(int i=0 ; i < num_render ; i++)
            {
                SGLFW_Render* _render = render[i] ; 
                _render->render(); 
            }
        }
        gl.renderloop_tail();      // swap buffers, poll events
    }
    exit(EXIT_SUCCESS);
}

