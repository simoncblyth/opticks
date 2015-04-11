#ifndef DEMO_H 
#define DEMO_H

#include "GMesh.hh"

class Demo : public GMesh {
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


#endif



