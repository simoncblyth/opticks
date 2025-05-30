#pragma once
/**
SGLFW_Event.h : manage scene data and OpenGL render pipelines
===============================================================

Yuxiang started from SGLFW_Scene and added rendering of record arrays
together with the OpenGL mesh based render.
The changes between SGLFW_Scene.h and SGLFW_Event.h
are not big enough to merit the duplication in the longterm.

Hence have extracted the non-duplicated into SGLGW_Evt.h
making it likely that can remove this.


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

#include "spath.h"
#include "SScene.h"
#include "SGLFW.h"

#include "SRecordInfo.h"
#include "SGLFW_Record.h"

struct SGLFW_Event
{
    SGLFW&        gl ;
    SGLM&         gm ;

    SGLFW_Record*   ar ;
    SGLFW_Record*   br ;

    const char* shader_fold ;
    SGLFW_Program* wire ;
    SGLFW_Program* iwire ;
    SGLFW_Program* norm ;
    SGLFW_Program* inorm ;

    SGLFW_Program*  rec_prog;

    std::vector<SGLFW_Mesh*> mesh ;

    SGLFW_Event(SGLFW& _gl );
    void init();
    void initProg();
    std::string  desc() const;

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




inline SGLFW_Event::SGLFW_Event(SGLFW& _gl )
    :
    gl(_gl),
    gm(gl.gm),
    ar(SGLFW_Record::Create(gm.ar, gm.timeparam_ptr)),
    br(SGLFW_Record::Create(gm.br, gm.timeparam_ptr)),
    shader_fold("${SGLFW_Event__shader_fold:-$OPTICKS_PREFIX/gl}"),
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
    wire = new SGLFW_Program(spath::Resolve(shader_fold,  "wireframe"), "vPos", "vNrm", nullptr, "MVP", gm.MVP_ptr );
    iwire = new SGLFW_Program(spath::Resolve(shader_fold, "iwireframe"), "vPos", "vNrm", "vInstanceTransform", "MVP", gm.MVP_ptr );

    norm = new SGLFW_Program(spath::Resolve(shader_fold, "normal"), "vPos", "vNrm", nullptr, "MVP", gm.MVP_ptr );
    inorm = new SGLFW_Program(spath::Resolve(shader_fold, "inormal"), "vPos", "vNrm", "vInstanceTransform", "MVP", gm.MVP_ptr );

    rec_prog = new SGLFW_Program(spath::Resolve(shader_fold, "rec_flying_point_persist"), nullptr, nullptr, nullptr, "ModelViewProjection", gm.MVP_ptr );
}


inline std::string  SGLFW_Event::desc() const
{
    std::stringstream ss;
    ss
        << "[SGLFW_Event::desc\n"
        << "  shader_fold [" << ( shader_fold ? shader_fold : "-" ) << "]\n"
        << "]SGLFW_Event::desc\n"
        ;
    std::string str = ss.str();
    return str ;
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
    int num_meshmerge = gm.scene->meshmerge.size();

    const std::vector<glm::tmat4x4<float>>& inst_tran = gm.scene->inst_tran ;
    const float* values = (const float*)inst_tran.data() ;
    int item_values = 4*4 ;

    for(int i=0 ; i < num_meshmerge ; i++)
    {
        const int4&  _inst_info = gm.scene->inst_info[i] ;

        int num_inst = _inst_info.y ;
        int offset   = _inst_info.z ;
        bool is_instanced = num_inst > 1 ;

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
    while(gl.renderloop_proceed())
    {
        gl.renderloop_head();  // clears
        render();
        gl.renderloop_tail();      // swap buffers, poll events
    }
}






