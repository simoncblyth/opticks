/**
examples/UseShaderSGLFW_SScene.cc
===================================

::

    ~/o/examples/UseShaderSGLFW_SScene/go.sh 

See also::

    TEST=CreateFromTree ~/o/sysrap/tests/SScene_test.sh 
    TEST=Load           ~/o/sysrap/tests/SScene_test.sh 

    ~/o/u4/tests/U4TreeCreateTest.sh 

    ~/o/u4/tests/U4Mesh_test.sh 
    ~/o/sysrap/tests/SMesh_test.sh 

**/

#include "ssys.h"
#include "SScene.h"
#include "SGLM.h"
#include "SGLFW.h"


int main()
{
    SScene* scene = SScene::Load("$SCENE_FOLD/scene"); 
    std::cout 
        << " scene " << scene->desc() 
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

    // HMM: could discover these from file system 
    SGLFW_Program prog("$SHADER_FOLD/wireframe", "vPos", "vNrm", nullptr, "MVP", gm.MVP_ptr ); 
    SGLFW_Program iprog("$SHADER_FOLD/iwireframe", "vPos", "vNrm", "vInstanceTransform", "MVP", gm.MVP_ptr ); 


    std::vector<SGLFW_Mesh*> mesh ; 
    int num_mesh_grup = scene->mesh_grup.size(); 

    const std::vector<glm::tmat4x4<float>>& inst_tran = scene->inst_tran ; 
    const float* values = (const float*)inst_tran.data() ; 
    int item_values = 4*4 ; 

    for(int i=0 ; i < num_mesh_grup ; i++)
    {
        //if( i != 4) continue ; 
        const int4&  _inst_info = scene->inst_info[i] ; 

        int num_inst = _inst_info.y ; 
        int offset   = _inst_info.z ; 
        bool is_instanced = _inst_info.y > 1 ; 

        const SMesh* _mg = scene->mesh_grup[i] ; 

        SGLFW_Mesh* _mesh = new SGLFW_Mesh(_mg);
        if( is_instanced )
        {
            _mesh->set_inst( num_inst, values + offset*item_values );  
            std::cout << _mesh->desc() << std::endl ; 
        }
        mesh.push_back(_mesh); 
    }
    int num_mesh = mesh.size(); 



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

            for(int i=0 ; i < num_mesh ; i++)
            {
                SGLFW_Mesh* _mesh = mesh[i] ; 
                SGLFW_Program* _prog = _mesh->has_inst() ? &iprog : &prog ;  
                _mesh->render(_prog);   
            }

        }
        gl.renderloop_tail();      // swap buffers, poll events
    }
    exit(EXIT_SUCCESS);
}

