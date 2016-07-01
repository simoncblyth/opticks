
#include "NGLM.hpp"
#include "Clipper.hh"
#include "ClipperCfg.hh"



template OKCORE_API void BCfg::addOptionF<Clipper>(Clipper*, const char*, const char* );
template OKCORE_API void BCfg::addOptionI<Clipper>(Clipper*, const char*, const char* );
template OKCORE_API void BCfg::addOptionS<Clipper>(Clipper*, const char*, const char* );




template <class Listener>
ClipperCfg<Listener>::ClipperCfg(const char* name, Listener* listener, bool live) 
      : 
       BCfg(name, live) 
{
       addOptionI<Listener>(listener, Listener::CUTPRINT, "Print debug info, when 1 is provided  ");
       addOptionI<Listener>(listener, Listener::CUTMODE,  "Integer cutmode control, -1 for disabled  ");
       addOptionS<Listener>(listener, Listener::CUTPOINT, "Comma delimited x,y,z clipping plane point in the plane in model-extent coordinates, eg 0,0,0  ");
       addOptionS<Listener>(listener, Listener::CUTNORMAL,"Comma delimited x,y,z clipping plane normal to the plane in model-extent coordinates, eg 1,0,0  ");
       addOptionS<Listener>(listener, Listener::CUTPLANE, "Comma delimited x,y,z,w world frame plane equation eg 1,0,0,16520  ");
}





template class OKCORE_API ClipperCfg<Clipper> ;



