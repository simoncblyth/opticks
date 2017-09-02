#pragma once

#include <string>


struct RBuf4 ; 
struct BBufSpec ; 
template <typename T> class NPY ; 

class GDrawable ; 
class GMergedMesh ; 

class Composition ; 

#include "RendererBase.hh"
#include "OGLRAP_API_EXPORT.hh"

/**
InstLODCull
=============

Provisioned from Scene, used from paired Renderer



**/


class OGLRAP_API InstLODCull : public RendererBase  {
  public:
      static const unsigned INSTANCE_MINIMUM ; 
      static const unsigned LOC_InstanceTransform ;

      InstLODCull(const char* tag, const char* dir=NULL, const char* incl_path=NULL);
      virtual ~InstLODCull();

      void upload(GMergedMesh* geometry, bool debug=false);
      void setComposition(Composition* composition);
      Composition* getComposition(); 

      bool isEnabled() const ;

      std::string desc() const ;

  private:
      void setITransformsBuffer(NPY<float>* ibuf);
      void initShader();

  private:
      Composition* m_composition ;
      GDrawable*   m_drawable ;
      GMergedMesh* m_geometry ;
      NPY<float>*  m_itransforms ; 
      unsigned     m_num_instance ; 
      bool         m_enabled ; 

      RBuf4*       m_fork ; 
 
   

};



