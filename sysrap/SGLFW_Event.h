#pragma once
/**
SGLFW_Event.h : manage scene data and OpenGL render pipelines
===============================================================

Yuxiang started from SGLFW_Scene and added rendering of record arrays
together with the OpenGL mesh based render.
The changes between SGLFW_Scene.h and SGLFW_Event.h
are not big enough to merit the duplication in the longterm.


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

#include "SRecordInfo.h"
#include "SGLFW_Record.h"

struct SGLFW_Event
{
    const SScene* sc ;
    SGLM&         gm ;
    SGLFW*        gl ;

    SGLFW_Record*   ar ;
    SGLFW_Record*   br ;

    SGLFW_Program* wire ;
    SGLFW_Program* iwire ;
    SGLFW_Program* norm ;
    SGLFW_Program* inorm ;

    SGLFW_Program*  rec_prog;

    std::vector<SGLFW_Mesh*> mesh ;

    SGLFW_Event(const SScene* scene, SGLM& gm );
    void init();
    void initProg();
    void initMesh();

    SGLFW_Program* getIProg() const ;
    SGLFW_Program* getProg() const ;

    void render();
    void render_mesh();
    void renderloop();
};


inline SGLFW_Program* SGLFW_Event::getIProg() const
{
    return gm.toggle.norm ? inorm : iwire ;
}
inline SGLFW_Program* SGLFW_Event::getProg() const
{
    return gm.toggle.norm ? norm : wire ;
}


inline SGLFW_Event::SGLFW_Event(const SScene* _sc, SGLM& _gm )
    :
    sc(_sc),
    gm(_gm),
    gl(new SGLFW(gm)),
    ar(SGLFW_Record::Create(gm.ar, gm.timeparam_ptr)),
    br(SGLFW_Record::Create(gm.br, gm.timeparam_ptr)),
    wire(nullptr),
    iwire(nullptr),
    norm(nullptr),
    inorm(nullptr),
    rec_prog(nullptr)
{
    init();
}

inline void SGLFW_Event::init()
{
    initProg();
    initMesh();
}



/**
SGLFW_Event::initProg
----------------------

Create the shaders

**/

inline void SGLFW_Event::initProg()
{
    // HMM: could discover these from file system
    wire = new SGLFW_Program("$SHADER_FOLD/wireframe", "vPos", "vNrm", nullptr, "MVP", gm.MVP_ptr );
    iwire = new SGLFW_Program("$SHADER_FOLD/iwireframe", "vPos", "vNrm", "vInstanceTransform", "MVP", gm.MVP_ptr );

    norm = new SGLFW_Program("$SHADER_FOLD/normal", "vPos", "vNrm", nullptr, "MVP", gm.MVP_ptr );
    inorm = new SGLFW_Program("$SHADER_FOLD/inormal", "vPos", "vNrm", "vInstanceTransform", "MVP", gm.MVP_ptr );


    rec_prog = new SGLFW_Program("$RECORDER_SHADER_FOLD", nullptr, nullptr, nullptr, "ModelViewProjection", gm.MVP_ptr );
    // there is no instanced variant of rec_prog
}

/**

SGLFW_Event::initMesh
-----------------------

Traverses the meshmerge vector from SScene
passing them to SGLFW_Mesh instances
which do the OpenGL uploads

HMM: how to match ray trace IAS/GAS handle selection ?

**/

inline void SGLFW_Event::initMesh()
{
    int num_meshmerge = sc->meshmerge.size();

    const std::vector<glm::tmat4x4<float>>& inst_tran = sc->inst_tran ;
    const float* values = (const float*)inst_tran.data() ;
    int item_values = 4*4 ;

    for(int i=0 ; i < num_meshmerge ; i++)
    {
        const int4&  _inst_info = sc->inst_info[i] ;

        int num_inst = _inst_info.y ;
        int offset   = _inst_info.z ;
        bool is_instanced = num_inst > 1 ;

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



/**
SGLFW_Event::render
--------------------

**/

inline void SGLFW_Event::render()
{
    if(      gm.option.M) render_mesh() ; 
    if(ar && gm.option.A) ar->render(rec_prog);   //record.npy render
    if(br && gm.option.B) br->render(rec_prog);   //record.npy render
}


/**
SGLFW_Event::render_mesh
-------------------------

Possibility: indirect OpenGL to avoid the draw loop
but while the number of meshes is small the 
motivation is not strong. 

Note the draw loop does have the advantage of
being able to use different shader pipeline
for different mesh (eg to highlight things).

**/

inline void SGLFW_Event::render_mesh()
{
    int num_mesh = mesh.size();
    for(int i=0 ; i < num_mesh ; i++)
    {
        bool viz = gm.is_vizmask_set(i);
        if(!viz) continue ;

        SGLFW_Mesh* _mesh = mesh[i] ;
        SGLFW_Program* _prog = _mesh->has_inst() ? getIProg() : getProg() ;
        _mesh->render(_prog);
    }
}



/**
SGLFW_Event::renderloop
------------------------

For ease of integration with alternative renders (eg raytrace)
it is often preferable to directly implement the renderloop
in the main or elsewhere and not use this simple renderloop.

**/

inline void SGLFW_Event::renderloop()
{
    while(gl->renderloop_proceed())
    {
        gl->renderloop_head();  // clears
        render();
        gl->renderloop_tail();      // swap buffers, poll events
    }
}

