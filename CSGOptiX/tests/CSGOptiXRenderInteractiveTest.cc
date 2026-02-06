/**
CSGOptiXRenderInteractiveTest.cc : Interactive raytrace rendering of analytic geometry
========================================================================================

Analytic CSGOptiX rendering with SGLM/SGLFW interactive control/visualization.
Usage with::

   ~/o/cx.sh
   ~/o/CSGOptiX/cxr_min.sh

Note this provides only CSGOptiX ray trace rendering, there is no rasterized toggle
with "C" key to switch between CUDA and OpenGL rendering.
For that functionality (but with triangulated geometry only) use:

   ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh
   ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc

TODO:

* navigation frames currently managed in SScene(tri) but they
  are equally relevant to ana+tri.

  * Relocate the navigation frames, where ? Consider the workflow.

* frame hopping like ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc

* interactive vizmask control to hide/show geometry

* WIP: enable mix and match ana/tri geometry, by incorporation
  of SScene and SOPTIX

* imgui reincarnation


**/

#include "ssys.h"
#include "stree.h"
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "SGLFW.h"
#include "SGLFW_CUDA.h"
#include "SGLFW_Evt.h"

#include "SScene.h"



struct CSGOptiXRenderInteractiveTest
{
    static constexpr const char* _level = "CSGOptiXRenderInteractiveTest__level" ;
    static constexpr const char* _ALLOW_REMOTE = "CSGOptiXRenderInteractiveTest__ALLOW_REMOTE" ;
    static int Initialize(bool allow_remote);

    int level ;
    bool ALLOW_REMOTE ;
    int irc ;

    SRecord*    ar ;
    SRecord*    br ;

    CSGFoundry* fd ;
    SGLM*       gm ;
    CSGOptiX*   cx ;

    SGLFW*      gl ;
    SGLFW_CUDA* interop ;   // holder of CUDA/OptiX buffer
    SGLFW_Evt*  glev ;

    CSGOptiXRenderInteractiveTest();

    void init();
    void initGeom();
    void initRecord();
    void initRender();

    void handle_snap_cx();
    void optix_render_to_buffer();
    void display_optix_buffer();
    void optix_render();

    std::string desc() const ;
};


inline int CSGOptiXRenderInteractiveTest::Initialize(bool allow_remote)
{
    SEventConfig::SetRGModeRender();
    bool is_remote_session = ssys::is_remote_session();
    if(is_remote_session && allow_remote == false )
    {
        std::cout << "CSGOptiXRenderInteractiveTest::Initialize : ABORTING : as detected remote session from SSH_TTY SSH_CLIENT \n";
        std::cout << "to override : export " << _ALLOW_REMOTE << "=1  ## warning have see gnome-shell crash with Wayland \n" ;
        return 1 ;
    }
    return 0 ;
}


inline CSGOptiXRenderInteractiveTest::CSGOptiXRenderInteractiveTest()
    :
    level(ssys::getenvint(_level,0)),
    ALLOW_REMOTE(ssys::getenvbool(_ALLOW_REMOTE)),
    irc(Initialize(ALLOW_REMOTE)),
    ar(SRecord::Load("$AFOLD", "$AFOLD_RECORD_SLICE")),
    br(SRecord::Load("$BFOLD", "$BFOLD_RECORD_SLICE")),
    fd(CSGFoundry::Load()),
    gm(new SGLM),
    cx(nullptr),
    gl(nullptr),
    interop(nullptr),
    glev(nullptr)
{
    init();
}

inline void CSGOptiXRenderInteractiveTest::init()
{
    initGeom();
    initRecord();
    initRender();
}

inline void CSGOptiXRenderInteractiveTest::initGeom()
{
    assert( irc == 0 );
    assert(fd);
    stree* tree = fd->getTree();
    assert(tree);
    SScene* scene = fd->getScene() ;
    assert(scene);
    gm->setTreeScene(tree, scene);
    gm->set_frame();   // MOI frame initially
}

inline void CSGOptiXRenderInteractiveTest::initRecord()
{
    gm->setRecord(ar, br);
}

inline void CSGOptiXRenderInteractiveTest::initRender()
{
    cx = CSGOptiX::Create(fd) ;
    gl = new SGLFW(*gm);
    interop = new SGLFW_CUDA(*gm);
    glev    = new SGLFW_Evt(*gl);

    if(gl->level > 0) std::cout << "CSGOptiXRenderInteractiveTest::initRender before render loop  gl.get_wanted_frame_idx " <<  gl->get_wanted_frame_idx() << "\n" ;
    if(level > 0) std::cout << "CSGOptiXRenderInteractiveTest::initRender [" << _level << "][" << level << "]\n" << desc() ;
}





/**
CSGOptiXRenderInteractiveTest::handle_snap_cx
----------------------------------------------

Saves ray trace geometry screenshots when certain keys pressed.
Formerly done between render_launch and unmap

**/

inline void CSGOptiXRenderInteractiveTest::handle_snap_cx()
{
    int wanted_snap = gl->get_wanted_snap();
    if(cx->handle_snap(wanted_snap)) gl->set_wanted_snap(0);
}

/**
CSGOptiXRenderInteractiveTest::optix_render_to_buffer
---------------------------------------------------------

1. interop.output_buffer::map : pass "baton" to CUDA/OptiX via d_pixels,
   device side pointer where to write kernel output

2. ray tracing OptiX launch populating the pixels

3. interop.output_buffer::unmap : pass baton back to OpenGL for display

**/

inline void CSGOptiXRenderInteractiveTest::optix_render_to_buffer()
{
    uchar4* d_pixels = interop->output_buffer->map() ;
    cx->setExternalDevicePixels(d_pixels);
    cx->render_launch();
    interop->output_buffer->unmap() ;
}

inline void CSGOptiXRenderInteractiveTest::display_optix_buffer()
{
    interop->displayOutputBuffer(gl->window);
}


inline void CSGOptiXRenderInteractiveTest::optix_render()
{
    handle_snap_cx();
    optix_render_to_buffer();
    display_optix_buffer();
}

inline std::string CSGOptiXRenderInteractiveTest::desc() const
{
    std::stringstream ss ;
    ss
        << "[CSGOptiXRenderInteractiveTest::desc\n"
        << " ar\n" << ( ar ? ar->desc() : "-" ) << "\n"
        << " br\n" << ( br ? br->desc() : "-" ) << "\n"
        << "]CSGOptiXRenderInteractiveTest::desc\n"
        ;
    std::string str = ss.str() ;
    return str ;
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    CSGOptiXRenderInteractiveTest t ;

    SGLFW* gl = t.gl ;

    while(gl->renderloop_proceed())
    {
        gl->renderloop_head();
        gl->handle_frame_hop();

        if(gl->gm.option.O) t.optix_render(); // alt-O toggle
        t.glev->render(); // alt-A/B toggle record array render

        gl->renderloop_tail();
    }
    return 0 ;
}

