/**
examples/UseShaderSGLFW_SScene_encapsulated.cc
===============================================

::

    ~/o/examples/UseShaderSGLFW_SScene_encapsulated/go.sh 

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

struct SGLFW_Scene
{
    const SScene* sc ; 
    sframe*       fr ; 
    SGLM*         gm ; 
    SGLFW*        gl ; 
#ifdef WITH_CUDA_GL_INTEROP 
    SGLFW_CUDA*   cu ; 
#endif
    // map of these ? or pairs ?
    SGLFW_Program* wire ; 
    SGLFW_Program* iwire ;
    SGLFW_Program* norm ; 
    SGLFW_Program* inorm ;

    std::vector<SGLFW_Mesh*> mesh ; 

    SGLFW_Scene(const SScene* scene ); 
    void init(); 
    void initGLM(); 
    void initProg(); 
    void initMesh(); 

    SGLFW_Program* getIProg() const ; 
    SGLFW_Program* getProg() const ; 

    void render(); 
    void renderloop(); 
}; 

inline SGLFW_Program* SGLFW_Scene::getIProg() const 
{
    return gl->toggle.norm ? inorm : iwire ;  
}
inline SGLFW_Program* SGLFW_Scene::getProg() const 
{
    return gl->toggle.norm ? norm : wire ;  
}


inline SGLFW_Scene::SGLFW_Scene(const SScene* _sc )
    :
    sc(_sc)
   ,fr(new sframe)
   ,gm(new SGLM)
   ,gl(new SGLFW(*gm))
#ifdef WITH_CUDA_GL_INTEROP 
   ,cu(new SGLFW_CUDA(*gm))  
#endif
   ,wire(nullptr)
   ,iwire(nullptr)
   ,norm(nullptr)
   ,inorm(nullptr)
{
    init(); 
}

inline void SGLFW_Scene::init()
{
    initGLM();
    initProg();
    initMesh();
}

inline void SGLFW_Scene::initGLM()
{
    fr->ce = make_float4(0.f, 0.f, 0.f, 1000.f); 
    gm->set_frame(*fr) ; 
    // TODO: SGLM::set_ce/set_fr ? avoid heavyweight sframe when not needed ?
}

inline void SGLFW_Scene::initProg()
{
    // HMM: could discover these from file system 
    wire = new SGLFW_Program("$SHADER_FOLD/wireframe", "vPos", "vNrm", nullptr, "MVP", gm->MVP_ptr ); 
    iwire = new SGLFW_Program("$SHADER_FOLD/iwireframe", "vPos", "vNrm", "vInstanceTransform", "MVP", gm->MVP_ptr ); 

    norm = new SGLFW_Program("$SHADER_FOLD/normal", "vPos", "vNrm", nullptr, "MVP", gm->MVP_ptr ); 
    inorm = new SGLFW_Program("$SHADER_FOLD/inormal", "vPos", "vNrm", "vInstanceTransform", "MVP", gm->MVP_ptr ); 
}

inline void SGLFW_Scene::initMesh()
{
    int num_mesh_grup = sc->mesh_grup.size(); 

    const std::vector<glm::tmat4x4<float>>& inst_tran = sc->inst_tran ; 
    const float* values = (const float*)inst_tran.data() ; 
    int item_values = 4*4 ; 

    for(int i=0 ; i < num_mesh_grup ; i++)
    {
        //if( i != 4) continue ; 
        const int4&  _inst_info = sc->inst_info[i] ; 

        int num_inst = _inst_info.y ; 
        int offset   = _inst_info.z ; 
        bool is_instanced = _inst_info.y > 1 ; 

        const SMesh* _mg = sc->mesh_grup[i] ; 

        SGLFW_Mesh* _mesh = new SGLFW_Mesh(_mg);
        if( is_instanced )
        {
            _mesh->set_inst( num_inst, values + offset*item_values );  
            std::cout << _mesh->desc() << std::endl ; 
        }
        mesh.push_back(_mesh); 
    }
}



inline void SGLFW_Scene::render()
{
    if( gl->toggle.cuda )
    {
#ifdef WITH_CUDA_GL_INTEROP 
        cu->fillOutputBuffer(); 
        cu->displayOutputBuffer(gl->window);
#endif
    }
    else
    {
        int num_mesh = mesh.size(); 
        for(int i=0 ; i < num_mesh ; i++)
        {
            SGLFW_Mesh* _mesh = mesh[i] ; 
            SGLFW_Program* _prog = _mesh->has_inst() ? getIProg() : getProg() ;  
            _mesh->render(_prog);   
        }
    }
}

inline void SGLFW_Scene::renderloop()
{
    while(gl->renderloop_proceed())
    {
        gl->renderloop_head();  // clears 
        render(); 
        gl->renderloop_tail();      // swap buffers, poll events
    }
}

int main()
{
    SScene* scene = SScene::Load("$SCENE_FOLD/scene"); 
    std::cout << " scene " << scene->desc() << std::endl ;
 
    SGLFW_Scene sc(scene) ; 
    sc.renderloop(); 

    exit(EXIT_SUCCESS);
}

