#pragma once
/**
SGLFW_Evt.h : manage event data and corresponding OpenGL progs
===============================================================

Start from SGLFW_Event.h and remove the geometry rendering,
want to do geometry rendering with SGLFW_Scene.h not duplicated
in here.

If succeed can get rid of the mixed _Event.h and rename this to _Event.h

**/

#include "spath.h"
#include "SGLFW.h"
#include "SGLFW_Record.h"

struct SGLFW_Evt
{
    SGLFW&        gl ;
    SGLM&         gm ;

    SGLFW_Record*   ar ;
    SGLFW_Record*   br ;

    const char* shader_fold ;
    const char* shader_dir ;

    SGLFW_Program*  rec_prog;

    SGLFW_Evt(SGLFW& _gl );

    void render();
    std::string desc() const ;
};


inline SGLFW_Evt::SGLFW_Evt(SGLFW& _gl )
    :
    gl(_gl),
    gm(gl.gm),
    ar(SGLFW_Record::Create(gm.ar, gm.timeparam_ptr)),
    br(SGLFW_Record::Create(gm.br, gm.timeparam_ptr)),
    shader_fold("${SGLFW_Evt__shader_fold:-$OPTICKS_PREFIX/gl}"),
    shader_dir(spath::Resolve(shader_fold, "rec_flying_point_persist")),
    rec_prog(new SGLFW_Program(shader_dir, nullptr, nullptr, nullptr, "ModelViewProjection", gm.MVP_ptr ))
{
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
        << "  shader_fold [" << ( shader_fold ? shader_fold : "-" ) << "]\n"
        << "  shader_dir  [" << ( shader_dir  ? shader_dir  : "-" ) << "]\n"
        << "]SGLFW_Evt::desc\n"
        ;
    std::string str = ss.str();
    return str ;
}


