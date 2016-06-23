#pragma once 

#include "GMesh.hh"
#include "OGLRAP_API_EXPORT.hh"

class OGLRAP_API Demo : public GMesh {
  private:
      static const float pvertex[] ;
      static const float pnormal[] ;
      static const float pcolor[] ;
      static const float ptexcoord[] ;
      static const unsigned int pindex[] ;
  public:
      Demo();
      virtual ~Demo();

};      





