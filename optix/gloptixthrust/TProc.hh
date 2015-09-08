#pragma once
#include "BufSpec.hh"

class TProc {
   public:
       TProc( BufSpec vtx ); 
   public:
       void check(); 
       void tscale(float factor); 
       void tgenerate(float radius); 
   private:
       BufSpec      m_vtx ; 
};

inline TProc::TProc(BufSpec vtx ) :
    m_vtx(vtx)
{
}


