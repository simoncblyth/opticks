#include "NGLM.hpp"
#include "View.hh"
#include "ViewCfg.hh"


template OKCORE_API void BCfg::addOptionF<View>(View*, const char*, const char* );
template OKCORE_API void BCfg::addOptionI<View>(View*, const char*, const char* );
template OKCORE_API void BCfg::addOptionS<View>(View*, const char*, const char* );





template <class Listener>
ViewCfg<Listener>::ViewCfg(const char* name, Listener* listener, bool live) 
   : 
   BCfg(name, live) 
{
       addOptionS<Listener>(listener, "eye", "Comma delimited eye position in model-extent coordinates, eg 0,0,-1  ");
       addOptionS<Listener>(listener, "look","Comma delimited look position in model-extent coordinates, eg 0,0,0  ");
       addOptionS<Listener>(listener, "up",  "Comma delimited up direction in model-extent frame, eg 0,1,0 " );
}




template class OKCORE_API ViewCfg<View> ;

