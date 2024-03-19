/**
examples/UseShaderSGLFW_SScene.cc
===================================

::

    ~/o/examples/UseShaderSGLFW_SScene/go.sh 

See also::

    ~/o/u4/tests/U4Mesh_test.sh 
    ~/o/sysrap/tests/SMesh_test.sh 
    ~/o/sysrap/tests/SScene_test.sh 

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


    SGLFW_Program prog("$SHADER_FOLD/wireframe", "vPos", "vNrm" ); 
    prog.use(); 
    prog.locateMVP("MVP",  gm.MVP_ptr );  

    SGLFW_Program iprog("$SHADER_FOLD/iwireframe", "vPos", "vNrm", "vInstanceTransform" ); 
    iprog.use(); 
    iprog.locateMVP("MVP", gm.MVP_ptr );  

    std::vector<SGLFW_Render*> render ; 
    int num_mesh_grup = scene->mesh_grup.size(); 

    const std::vector<glm::tmat4x4<float>>& inst_tran = scene->inst_tran ; 
    const float* values = (const float*)inst_tran.data() ; 
    int itemsize = 4*4*sizeof(float) ; 

    for(int i=0 ; i < num_mesh_grup ; i++)
    {
        if( i != 3) continue ; 

        const int4&  _inst_info = scene->inst_info[i] ; 

        int num_inst = _inst_info.y ; 
        int offset   = _inst_info.z ; 

        const SMesh* _mesh = scene->mesh_grup[i] ; 
        SGLFW_Program* _prog = &prog ;  

        SGLFW_Render* _render = new SGLFW_Render(_mesh, _prog );
        if( _inst_info.y > 1 )
        {
            _render->set_inst( num_inst, values + offset*itemsize );  
        }
  
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

