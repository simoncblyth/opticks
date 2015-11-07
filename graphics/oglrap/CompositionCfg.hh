#pragma once
#include "Cfg.hh"

template <class Listener>
class CompositionCfg : public Cfg {
public:
   CompositionCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {
       addOptionI<Listener>(listener, Listener::PRINT,    "Print");
       addOptionS<Listener>(listener, Listener::SELECT,   "Selection, four comma delimited integers");
       addOptionS<Listener>(listener, Listener::PICKPHOTON, 
           "[UDP only], up to 4 comma delimited integers, eg:\n"
           "10000   : target view at the center extent \n" 
           "10000,1 : as above but hide other records \n" 
           "\n"
           "see CompositionCfg.hh\n"
      );

       addOptionS<Listener>(listener, Listener::PICKFACE, 
           "[UDP only], up to 4 comma delimited integers, eg 10,11,3158,0  \n"
           "to target single face index 10 (range 10:11) of solid index 3158 in mesh index 0 \n" 
           "\n"
           "    face_index0 \n" 
           "    face_index1 \n" 
           "    solid_index \n" 
           "    mergedmesh_index  (currently only 0 non-instanced operational) \n" 
           "\n"
           "see: CompositionCfg.hh\n"
           "     Composition::setPickFace\n"
           "     Scene::setFaceRangeTarget\n"
           "     GGeo::getFaceRangeCenterExtent\n"
      );

   }
};


