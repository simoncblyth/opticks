#include "NGLM.hpp"
#include "Trackball.hh"
#include "TrackballCfg.hh"

template OKCORE_API void BCfg::addOptionF<Trackball>(Trackball*, const char*, const char* );
template OKCORE_API void BCfg::addOptionI<Trackball>(Trackball*, const char*, const char* );
template OKCORE_API void BCfg::addOptionS<Trackball>(Trackball*, const char*, const char* );




template <class Listener>
TrackballCfg<Listener>::TrackballCfg(const char* name, Listener* listener, bool live) 
    : 
    BCfg(name, live) 
{
       addOptionF<Listener>(listener, Listener::RADIUS,          "Trackball radius");
       addOptionF<Listener>(listener, Listener::TRANSLATEFACTOR, "Translation factor");

       addOptionS<Listener>(listener, Listener::ORIENTATION,     "Comma delimited theta,phi in degress");
       addOptionS<Listener>(listener, Listener::TRANSLATE,       "Comma delimited x,y,z translation triplet");
}





template class OKCORE_API TrackballCfg<Trackball> ;

