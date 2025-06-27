#pragma once
/**
SGLFW_Evt.h : manage event data and corresponding OpenGL progs
===============================================================

Started from the old SGLFW_Event.h and removed the geometry rendering,
as prefer modular arrangement with geometry rendering in SGLFW_Scene.h
and event rendering here.

::

    (ok) A[blyth@localhost opticks]$ opticks-fl SGLFW_Evt
    ./CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc
    ./sysrap/CMakeLists.txt
    ./sysrap/SRecord.h
    ./sysrap/SGLFW_Event_TO_BE_REMOVED.h
    ./sysrap/tests/SGLFW_SOPTIX_Scene_test.cc
    ./sysrap/tests/SGLFW_Evt_test.cc
    ./sysrap/tests/SGLFW_Evt_test.sh
    ./sysrap/tests/tests.txt
    ./sysrap/SGLFW_Evt.h


**/

#include "ssys.h"
#include "spath.h"
#include "SGLFW.h"
#include "SGLFW_Record.h"

struct SGLFW_Evt
{
    static constexpr const char* _SGLFW_Evt__level = "SGLFW_Evt__level" ;
    int           level ;

    SGLFW&        gl ;
    SGLM&         gm ;

    SGLFW_Record*   ar ;
    SGLFW_Record*   br ;

    const char* shader_fold ;
    const char* shader_name ;
    const char* shader_dir ;

    SGLFW_Program*  rec_prog;

    SGLFW_Evt(SGLFW& _gl );
    void init();

    void render();
    std::string desc() const ;
};


inline SGLFW_Evt::SGLFW_Evt(SGLFW& _gl )
    :
    level(ssys::getenvint(_SGLFW_Evt__level,0)),
    gl(_gl),
    gm(gl.gm),
    ar(SGLFW_Record::Create(gm.ar, gm.timeparam_ptr)),
    br(SGLFW_Record::Create(gm.br, gm.timeparam_ptr)),
    shader_fold(nullptr),
    shader_name(nullptr),
    shader_dir(nullptr),
    rec_prog(nullptr)
{
    init();
}

inline void SGLFW_Evt::init()
{
    shader_fold = spath::Resolve("${SGLFW_Evt__shader_fold:-$OPTICKS_PREFIX/gl}") ;
    shader_name = spath::Resolve("${SGLFW_Evt__shader_name:-rec_flying_point_persist}") ;
    shader_dir = spath::Resolve(shader_fold, shader_name);
    rec_prog = new SGLFW_Program(shader_dir, nullptr, nullptr, nullptr, "ModelViewProjection", gm.MVP_ptr );

    if(level>0) std::cout << desc();
}

/**
SGLFW_Evt::render
--------------------

**/

inline void SGLFW_Evt::render()
{
    if(ar && gm.option.A) ar->render(rec_prog);   //record.npy render
    if(br && gm.option.B) br->render(rec_prog);   //record.npy render
}


inline std::string  SGLFW_Evt::desc() const
{
    std::stringstream ss;
    ss
        << "[SGLFW_Evt::desc\n"
        << "  level " << level << "\n"
        << "  shader_fold [" << ( shader_fold ? shader_fold : "-" ) << "]\n"
        << "  shader_name [" << ( shader_name ? shader_name : "-" ) << "]\n"
        << "  shader_dir  [" << ( shader_dir  ? shader_dir  : "-" ) << "]\n"
        << "  rec_prog    [" << ( rec_prog ? "YES" : "NO ") << "]\n"
        << "]SGLFW_Evt::desc\n"
        ;
    std::string str = ss.str();
    return str ;
}


