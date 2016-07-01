
#include "NGLM.hpp"
#include "Camera.hh"
#include "CameraCfg.hh"



template OKCORE_API void BCfg::addOptionF<Camera>(Camera*, const char*, const char* );
template OKCORE_API void BCfg::addOptionI<Camera>(Camera*, const char*, const char* );
template OKCORE_API void BCfg::addOptionS<Camera>(Camera*, const char*, const char* );




template <class Listener>
CameraCfg<Listener>::CameraCfg(const char* name, Listener* listener, bool live) 
    : 
    BCfg(name, live) 
{
       addOptionI<Listener>(listener, Listener::PRINT,    "Print");

       addOptionF<Listener>(listener, Listener::ZOOM,     "Zoom factor");
       addOptionF<Listener>(listener, Listener::SCALE,    "Screen Scale");
       addOptionF<Listener>(listener, Listener::NEAR_,     "Near distance");
       addOptionF<Listener>(listener, Listener::FAR_,      "Far distance" );
       addOptionF<Listener>(listener, Listener::PARALLEL, "Parallel or perspective");
}


template class OKCORE_API CameraCfg<Camera> ;
