#pragma once
/**
CSGGenstep.h : Creator of CenterExtent "CE" Gensteps used by CSGOptiXSimtraceTest
==================================================================================

Center extent gensteps (maybe rename as FrameGensteps for simplicity without CE jargon) 
can generate for example photons in the local frame of part of the geometry in order 
to create 2D cross-sections from the intersects.  

Sensitive to envvars:

CEGS
   center-extent-genstep
   expect 4 or 7 ints delimited by colon nx:ny:nz:num_pho OR nx:px:ny:py:nz:py:num_pho 

TODO: compare with SCenterExtentGenstep.hh 

TODO: relocate 
   this is misplaced up in CSG, most of it should be down in SysRap/SEvent 
   to make ce-gensteps follow the pattern of other gensteps  
   perhaps using new "SLocation/SComposition/SFrame" struct (that is persistable)
   for holding transforms and ce that is provided by a CSGFoundry method
   and which is passed down to SysRap/SEvent to generate the gensteps 

For example::

    SEvent::MakeTorchGensteps()
    SEvent::MakeCarrierGensteps()

    SEvent::MakeFrameGensteps(const SFrame& fr)

**/

#include <vector>
#include "plog/Severity.h"

struct float4 ; 
struct qat4 ; 
template <typename T> struct Tran ;  

#include "CSG_API_EXPORT.hh"


struct NP ; 

struct CSG_API CSGGenstep
{
    CSGGenstep( const CSGFoundry* foundry );  
    void create(const char* moi, bool ce_offset, bool ce_scale );
    void generate_photons_cpu();
    void save(const char* basedir) const ; 

    // below are "private"

    static const plog::Severity LEVEL ; 
    void init(); 
    void locate(const char* moi); 
    void override_locate() ; 
    void configure_grid() ; 

    const CSGFoundry* foundry ; 
    float gridscale ;  
    const char* moi ; 
    int midx ; 
    int mord ; 
    int iidx ; 
    float4 ce ;
    qat4*  m2w ;              // 4x4 float
    qat4*  w2m ;              // 4x4 float 

    Tran<double>* geotran ;   // internally holds 3*glm::tmat4x4<double>
    // derived from m2w w2m using Tran<double>::FromPair 

    std::vector<int> cegs ; 

    NP* gs ; 
    NP* pp ; 


}; 
