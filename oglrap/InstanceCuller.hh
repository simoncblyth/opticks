#pragma once

#include <cstddef>
#include <vector>

struct BBufSpec ; 
template <typename T> class NPY ; 

class Composition ; 

#include "RendererBase.hh"
#include "OGLRAP_API_EXPORT.hh"

class OGLRAP_API InstanceCuller : public RendererBase  {
  public:
      InstanceCuller(const char* tag, const char* dir=NULL, const char* incl_path=NULL);
      virtual ~InstanceCuller();

      void setComposition(Composition* composition);
      Composition* getComposition(); 
  private:
      Composition* m_composition ;
  

};



