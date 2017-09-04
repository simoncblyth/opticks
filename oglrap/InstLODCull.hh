#pragma once

#include <string>


struct RContext ; 
struct RBuf ; 
struct RBuf4 ; 
class Composition ; 

#include "NGLM.hpp"

#include "RendererBase.hh"
#include "OGLRAP_API_EXPORT.hh"

/**
InstLODCull
=============

Provisioned from Scene, used from paired Renderer


Based on demo code developments in 
/Users/blyth/env/graphics/opengl/instcull/LODCullShader.cc

**/


class OGLRAP_API InstLODCull : public RendererBase  
{
       //friend class Renderer ; 
    public:
       static const unsigned QSIZE ;
       static const unsigned INSTANCE_MINIMUM ; 
       enum { LOD_MAX = 4 } ; 
       static const unsigned LOC_InstanceTransform ;

       InstLODCull(RContext* context, const char* tag, const char* dir=NULL, const char* incl_path=NULL);
       virtual ~InstLODCull();

       void setComposition(Composition* composition);
       std::string desc() const ;

       void setupFork(RBuf* src, RBuf4* dst,  RBuf4* dst_devnull);
       void launch();
  private:
       GLuint createForkVertexArray(RBuf* src, RBuf4* dst) ;
       void initShader();
       void check_uniforms();
       void update_uniforms();
       void applyFork();
       void applyForkStreamQueryWorkaround();
       void pullback();
  private:
       RContext*    m_context ; 
       Composition* m_composition ;
       RBuf*        m_src ; 
       RBuf4*       m_dst ; 
       RBuf4*       m_dst_devnull ; 

       unsigned     m_num_instance ; 
       unsigned     m_num_lod ; 
       unsigned     m_launch_count ; 
       
 
       GLuint       m_lodQuery[LOD_MAX] ; 
       GLuint       m_forkVAO ;  
       GLuint       m_workaroundVAO ;  
 
       glm::vec4    m_lodcut ;    // hmm if lowest LOD dist is less than near, never get to see the best level 


       //GLint  m_mv_location ;
       //GLint  m_mvp_location ;
       //GLint  m_lodcut_location ; 
     



};



