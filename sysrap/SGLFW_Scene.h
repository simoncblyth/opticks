#pragma once
/**
SGLFW_Scene.h : manage scene data and OpenGL render pipelines 
===============================================================

Primary members
-----------------

(SScene)sc
    source of SMesh data  
(SGLM)gm
     view/projection maths using glm 
(SGLFW)gl
     OpenGL render top level 


SGLFW_Program render pipelines
--------------------------------

wire
    wireframe
iwire
    instanced wireframe
norm
    normal shader
inorm
    instanced normal shader


**/

#include "SScene.h"
#include "SGLFW.h"

struct SGLFW_Scene
{
    static constexpr const char* _DUMP = "SGLFW_Scene__DUMP" ; 
    bool            DUMP ; 
    
    SGLFW&        gl ; 
    SGLM&         gm ; 

    const char* shader_fold ;
    SGLFW_Program* wire ; 
    SGLFW_Program* iwire ;
    SGLFW_Program* norm ; 
    SGLFW_Program* inorm ;

    std::vector<SGLFW_Mesh*> mesh ; 

    SGLFW_Scene(SGLFW& gl ); 
    void init(); 
    void initProg(); 
    void initMesh(); 

    SGLFW_Program* getIProg() const ; 
    SGLFW_Program* getProg() const ; 

    void render(); 
    void renderloop(); 
}; 



inline SGLFW_Program* SGLFW_Scene::getIProg() const 
{
    return gm.toggle.norm ? inorm : iwire ;  
}
inline SGLFW_Program* SGLFW_Scene::getProg() const 
{
    return gm.toggle.norm ? norm : wire ;  
}


inline SGLFW_Scene::SGLFW_Scene(SGLFW& _gl)
    :
    DUMP(ssys::getenvbool(_DUMP)), 
    gl(_gl),
    gm(gl.gm),
    shader_fold("${SGLFW_Scene__shader_fold:-$OPTICKS_PREFIX/gl}"),
    wire(nullptr),
    iwire(nullptr),
    norm(nullptr),
    inorm(nullptr)
{
    init(); 
}

inline void SGLFW_Scene::init()
{
    initProg();
    initMesh();
}

/**
SGLFW_Scene::initProg
----------------------

Create the shaders 

wire
   wireframe

iwire
   instanced wireframe

norm
   normal shader

inorm
   instanced normal shader
   

**/

inline void SGLFW_Scene::initProg()
{
    wire = new SGLFW_Program(spath::Resolve(shader_fold,  "wireframe"), "vPos", "vNrm", nullptr, "MVP", gm.MVP_ptr );
    iwire = new SGLFW_Program(spath::Resolve(shader_fold, "iwireframe"), "vPos", "vNrm", "vInstanceTransform", "MVP", gm.MVP_ptr );

    norm = new SGLFW_Program(spath::Resolve(shader_fold, "normal"), "vPos", "vNrm", nullptr, "MVP", gm.MVP_ptr );
    inorm = new SGLFW_Program(spath::Resolve(shader_fold, "inormal"), "vPos", "vNrm", "vInstanceTransform", "MVP", gm.MVP_ptr );
}

/**
SGLFW_Scene::initMesh
-----------------------

Traverses the meshmerge vector from SScene
passing them to SGLFW_Mesh instances 
which do the OpenGL uploads

HMM: how to match ray trace IAS/GAS handle selection ?

**/

inline void SGLFW_Scene::initMesh()
{
    int num_meshmerge = gm.scene->meshmerge.size(); 

    const std::vector<glm::tmat4x4<float>>& inst_tran = gm.scene->inst_tran ; 
    const float* values = (const float*)inst_tran.data() ; 
    int item_values = 4*4 ; 

    if(DUMP) std::cout 
         << "SGLFW_Scene::initMesh"
         << " num_meshmerge " << num_meshmerge
         << "\n"
         ;

    for(int i=0 ; i < num_meshmerge ; i++)
    {
        const int4&  _inst_info = gm.scene->inst_info[i] ; 

        int num_inst = _inst_info.y ; 
        int offset   = _inst_info.z ; 
        bool is_instanced = num_inst > 1 ; 

        if(DUMP) std::cout 
             << "SGLFW_Scene::initMesh"
             << " i " << i
             << " num_inst (inst_info.y) " << num_inst
             << " offset   (inst_info.z) " << offset 
             << " is_instanced " << ( is_instanced ? "YES" : "NO " ) 
             << "\n"
             ;

        const SMesh* _mm = gm.scene->meshmerge[i] ; 

        SGLFW_Mesh* _mesh = new SGLFW_Mesh(_mm);
        if( is_instanced )
        {
            _mesh->set_inst( num_inst, values + offset*item_values );  
            //std::cout << _mesh->desc() << std::endl ; 
        }
        mesh.push_back(_mesh); 
    }
}



/**
SGLFW_Scene::render
--------------------

TODO: indirect OpenGL to avoid the draw loop 

Note the draw loop does have the advantage of 
being able to use different shader pipeline 
for different mesh (eg to highlight things). 

TODO: VIZMASK for flexible skipping 

**/

inline void SGLFW_Scene::render()
{
    int num_mesh = mesh.size(); 

    if(DUMP) std::cout 
         << "SGLFW_Scene::render"
         << " num_mesh " << num_mesh
         << "\n"
         ;

    for(int i=0 ; i < num_mesh ; i++)
    {
        bool viz = gm.is_vizmask_set(i); 

        if(DUMP) std::cout 
             << "SGLFW_Scene::render"
             << " i " << i
             << " viz " << ( viz ? "YES" : "NO " )
             << "\n"
             ;

        if(!viz) continue ; 

        SGLFW_Mesh* _mesh = mesh[i] ; 
        bool has_inst = _mesh->has_inst() ; 

        if(DUMP) std::cout 
             << "SGLFW_Scene::render"
             << " i " << i
             << " has_inst " << ( has_inst ? "YES" : "NO " )
             << "\n"
             ;

        SGLFW_Program* _prog = has_inst ? getIProg() : getProg() ;  
        _mesh->render(_prog);   
    }
}


/**
SGLFW_Scene::renderloop
------------------------

For ease of integration with alternative renders (eg raytrace)
it is often preferable to directly implement the renderloop
in the main or elsewhere and not use this simple renderloop.

**/

inline void SGLFW_Scene::renderloop()
{
    while(gl.renderloop_proceed())
    {
        gl.renderloop_head();  // clears 
        render(); 
        gl.renderloop_tail();      // swap buffers, poll events
    }
}

