#pragma once
#include "Cfg.hh"

template <class Listener>
class GGeoCfg : public Cfg {
public:
   GGeoCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {

       addOptionS<Listener>(listener, Listener::PICKFACE, 
           "[UDP only], up to 4 comma delimited integers, eg 10,11,3158,0  \n"
           "to target single face index 10 (range 10:11) of solid index 3158 in mesh index 0 \n" 
           "\n"
           "    face_index0 \n" 
           "    face_index1 \n" 
           "    solid_index \n" 
           "    mergedmesh_index  (currently only 0 non-instanced operational) \n" 
           "\n"
           "see: GGeoCfg.hh\n"
           "     Composition::setPickFace\n"
           "     Scene::setFaceRangeTarget\n"
           "     GGeo::getFaceRangeCenterExtent\n"
      );

   }
};


