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

* WIP: moved some common CSGOptiX stuff down to sysrap header-only such as jpg/png writing and image annotation

* TODO: composite OpenGL event representation pixels together with OptiX ray traced renders

**/

#include "ssys.h"
#include "stree.h"
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "SGLFW.h"
#include "SGLFW_CUDA.h"

#include "SScene.h"



struct CSGOptiXRenderInteractiveTest
{
    static constexpr const char* _ALLOW_REMOTE = "CSGOptiXRenderInteractiveTest__ALLOW_REMOTE" ;
    static constexpr const char* _FRAME_HOP = "CSGOptiXRenderInteractiveTest__FRAME_HOP" ;
    static constexpr const char* _SGLM_DESC = "CSGOptiXRenderInteractiveTest__SGLM_DESC" ;
    static constexpr const char* _EXTENT_PFX = "EXTENT:" ;

    static int Initialize(bool allow_remote);
    static sfr MOI_Frame(float _extent, stree* _st, const char* MOI);

    bool ALLOW_REMOTE ;
    bool FRAME_HOP ;
    bool SGLM_DESC ;
    const char* MOI ;
    int irc ;

    CSGFoundry* fd ;
    SScene*     scene ;
    stree*      st ;
    float       extent ;
    sfr         mfr ;
    CSGOptiX*   cx ;

    SGLM& gm ;
    SGLFW gl ;
    SGLFW_CUDA interop ;   // holder of CUDA/OptiX buffer

    CSGOptiXRenderInteractiveTest();
    void init();
    void handle_frame_hop();
    void handle_snap();
    void optix_render_to_buffer();
    void display_buffer();

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

inline sfr CSGOptiXRenderInteractiveTest::MOI_Frame(float _extent, stree* _st, const char* MOI)
{
    assert( _st );
    sfr mfr = _extent > 0.f ? sfr::MakeFromExtent<float>(_extent) :  _st->get_frame(MOI);    // HMM: what about when start from CSGMaker geometry ?
    mfr.set_idx(-2);                 // maybe should start from stree/snode/sn geometry with an streemaker.h ?
    return mfr ;
}


inline CSGOptiXRenderInteractiveTest::CSGOptiXRenderInteractiveTest()
    :
    ALLOW_REMOTE(ssys::getenvbool(_ALLOW_REMOTE)),
    FRAME_HOP(ssys::getenvbool(_FRAME_HOP)),
    SGLM_DESC(ssys::getenvbool(_SGLM_DESC)),
    MOI(ssys::getenvvar("MOI", "0:0:-1")),   // default lvid 0 in remainder
    irc(Initialize(ALLOW_REMOTE)),
    fd(CSGFoundry::Load()),
    scene(fd ? fd->getScene() : nullptr),
    st( fd ? fd->getTree() : nullptr),
    extent( sstr::StartsWith(MOI, _EXTENT_PFX) ? sstr::To<float>( MOI + strlen(_EXTENT_PFX) ) : 0.f ),
    mfr(MOI_Frame(extent, st, MOI)),
    cx( fd ? CSGOptiX::Create(fd) : nullptr ),
    gm(*(cx->sglm)),
    gl(gm),
    interop(gm)
{
    init();
}

inline void CSGOptiXRenderInteractiveTest::init()
{
    assert( irc == 0 );
    assert(fd);
    assert(scene);
    assert(st);

    if(gl.level > 0) std::cout << "CSGOptiXRenderInteractiveTest::init before render loop  gl.get_wanted_frame_idx " <<  gl.get_wanted_frame_idx() << "\n" ;
}

/**
CSGOptiXRenderInteractiveTest::handle_frame_hop
-------------------------------------------------

When gl sees frame hop keypress get the frame and pass that to gm

**/


inline void CSGOptiXRenderInteractiveTest::handle_frame_hop()
{
    if(!FRAME_HOP) return ;

    int wanted_frame_idx = gl.get_wanted_frame_idx() ; // -2 until press number key 0-9, back to -2 when press M
    bool frame_hop = !gm.has_frame_idx(wanted_frame_idx) ;
    if(frame_hop)
    {
        if(gl.level > 0) std::cout << "main:" << _FRAME_HOP << " wanted_frame_idx: " << wanted_frame_idx << "\n";
        if( wanted_frame_idx == -2 )
        {
            gm.set_frame(mfr);
            if(SGLM_DESC) std::cout << _SGLM_DESC << "\n"  << gm.desc() ;
        }
        else if( wanted_frame_idx >= 0 )
        {
            sfr wfr = scene->getFrame(wanted_frame_idx) ;
            gm.set_frame(wfr);
        }
    }
}


/**
CSGOptiXRenderInteractiveTest::handle_snap
---------------------------------------------

Saves ray trace geometry screenshots when certain keys pressed.
Formerly done between render_launch and unmap

**/

inline void CSGOptiXRenderInteractiveTest::handle_snap()
{
    int wanted_snap = gl.get_wanted_snap();
    if( wanted_snap == 1 || wanted_snap == 2 )
    {
        std::cout << "CSGOptiXRenderInteractiveTest::handle_snap gl.get_wanted_snap calling cx->render_snap \n" ;
        switch(wanted_snap)
        {
            case 1: cx->render_save()          ; break ;
            case 2: cx->render_save_inverted() ; break ;
        }
        gl.set_wanted_snap(0);
    }
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
    uchar4* d_pixels = interop.output_buffer->map() ;
    cx->setExternalDevicePixels(d_pixels);
    cx->render_launch();
    interop.output_buffer->unmap() ;
}

inline void CSGOptiXRenderInteractiveTest::display_buffer()
{
    interop.displayOutputBuffer(gl.window);
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    CSGOptiXRenderInteractiveTest t ;
    SGLFW& gl = t.gl ;

    while(gl.renderloop_proceed())
    {
        gl.renderloop_head();
        t.handle_frame_hop();
        t.handle_snap();

        t.optix_render_to_buffer();

        t.display_buffer();
        gl.renderloop_tail();
    }
    return 0 ;
}

