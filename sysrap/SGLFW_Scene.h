#pragma once
/**
SGLFW_Scene.h
==============


**/

#include "SScene.h"
#include "SGLFW.h"

struct SGLFW_Scene
{
    static int RenderLoop(const SScene* scene, SGLM* gm ); 
    
    const SScene* sc ; 
    SGLM*         gm ; 
    SGLFW*        gl ; 

    // map of these ? or pairs ?
    SGLFW_Program* wire ; 
    SGLFW_Program* iwire ;
    SGLFW_Program* norm ; 
    SGLFW_Program* inorm ;

    std::vector<SGLFW_Mesh*> mesh ; 

    SGLFW_Scene(const SScene* scene, SGLM* gm ); 
    void init(); 
    void initProg(); 
    void initMesh(); 

    void setFrameIdx(int idx);

    SGLFW_Program* getIProg() const ; 
    SGLFW_Program* getProg() const ; 

    void render(); 
    void renderloop(); 
}; 

inline int SGLFW_Scene::RenderLoop(const SScene* scene, SGLM* gm ) // static
{
    std::cout << "[ SGLFW_Scene::RenderLoop" << std::endl << scene->desc() << std::endl ;
    SGLFW_Scene sc(scene, gm) ;
    sc.renderloop();  
    std::cout << "] SGLFW_Scene::RenderLoop" << std::endl << scene->desc() << std::endl ;
    return 0 ; 
}

inline SGLFW_Program* SGLFW_Scene::getIProg() const 
{
    return gl->toggle.norm ? inorm : iwire ;  
}
inline SGLFW_Program* SGLFW_Scene::getProg() const 
{
    return gl->toggle.norm ? norm : wire ;  
}


inline SGLFW_Scene::SGLFW_Scene(const SScene* _sc, SGLM* _gm)
    :
    sc(_sc)
   ,gm(_gm)
   ,gl(new SGLFW(*gm))
   ,wire(nullptr)
   ,iwire(nullptr)
   ,norm(nullptr)
   ,inorm(nullptr)
{
    init(); 
}

inline void SGLFW_Scene::init()
{
    initProg();
    initMesh();
}



inline void SGLFW_Scene::initProg()
{
    // HMM: could discover these from file system 
    wire = new SGLFW_Program("$SHADER_FOLD/wireframe", "vPos", "vNrm", nullptr, "MVP", gm->MVP_ptr ); 
    iwire = new SGLFW_Program("$SHADER_FOLD/iwireframe", "vPos", "vNrm", "vInstanceTransform", "MVP", gm->MVP_ptr ); 

    norm = new SGLFW_Program("$SHADER_FOLD/normal", "vPos", "vNrm", nullptr, "MVP", gm->MVP_ptr ); 
    inorm = new SGLFW_Program("$SHADER_FOLD/inormal", "vPos", "vNrm", "vInstanceTransform", "MVP", gm->MVP_ptr ); 
}

/**
SGLFW_Scene::initMesh
-----------------------

Traverses the mesh_group from the SScene
passing them to SGLFW_Mesh instances 
which do the OpenGL uploads

HMM: how to match ray trace IAS/GAS handle selection ?

**/

inline void SGLFW_Scene::initMesh()
{
    int num_meshmerge = sc->meshmerge.size(); 

    const std::vector<glm::tmat4x4<float>>& inst_tran = sc->inst_tran ; 
    const float* values = (const float*)inst_tran.data() ; 
    int item_values = 4*4 ; 

    for(int i=0 ; i < num_meshmerge ; i++)
    {
        //if( i != 4) continue ; 
        const int4&  _inst_info = sc->inst_info[i] ; 

        int num_inst = _inst_info.y ; 
        int offset   = _inst_info.z ; 
        bool is_instanced = _inst_info.y > 1 ; 

        const SMesh* _mm = sc->meshmerge[i] ; 

        SGLFW_Mesh* _mesh = new SGLFW_Mesh(_mm);
        if( is_instanced )
        {
            _mesh->set_inst( num_inst, values + offset*item_values );  
            //std::cout << _mesh->desc() << std::endl ; 
        }
        mesh.push_back(_mesh); 
    }
}



inline void SGLFW_Scene::setFrameIdx(int _idx)
{
    int num_frame = sc->frame.size(); 
    int idx = ( _idx > -1 && _idx < num_frame ) ? _idx : -1 ; 

    const float* _ce = sc->get_ce(0) ; 
    std::cout 
         << "SGLFW_Scene::setFrameIdx"
         << " num_frame " << num_frame 
         << " _idx " << _idx
         << " idx " << idx
         << " _ce[3] " << ( _ce ? _ce[3] : -1.f )    
         << "\n" 
         ; 
  
    sfr fr = idx == -1 ? sfr::MakeFromCE(_ce) : sc->frame[idx] ; 
    fr.set_idx(idx); 
    gm->set_frame(fr);

    assert( gm->get_frame_idx() == idx ); 
}



/**
SGLFW_Scene::render
--------------------

TODO: indirect OpenGL to avoid the draw loop 

Note the draw loop does have the advantage of 
being able to use different shader pipeline 
for different mesh (eg to highlight things). 

**/

inline void SGLFW_Scene::render()
{
    int num_mesh = mesh.size(); 
    for(int i=0 ; i < num_mesh ; i++)
    {
        SGLFW_Mesh* _mesh = mesh[i] ; 
        SGLFW_Program* _prog = _mesh->has_inst() ? getIProg() : getProg() ;  
        _mesh->render(_prog);   
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

