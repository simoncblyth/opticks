#pragma once
#include "CBufSpec.hh"

class TProc {
   public:
       TProc( CBufSpec vtx ); 
   public:
       void check(); 
       void tscale(float factor); 
       void tgenerate(float radius); 
   private:
       CBufSpec      m_vtx ; 
};

inline TProc::TProc(CBufSpec vtx ) :
    m_vtx(vtx)
{
}


