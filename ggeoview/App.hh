#pragma once

class Opticks ;    // okc-
class OpticksHub ; // opticksgeo-
class OpticksIdx ; // opticksgeo-

#ifdef WITH_OPTIX
class OpEngine ;   // opop-
class OpViz ;      // optixgl-
#endif

class OpticksViz ; // ggeoview-


#include "GGV_API_EXPORT.hh"
#include "GGV_HEAD.hh"


// NB ####### THIS IS BEING REPLACED WITH OpticksMgr  ##########

class GGV_API App {
  public:
       App(int argc, char** argv );
       void configure();
  private:
       void init(int argc, char** argv);
  public:
       void prepareViz();   // creates OpenGL context window and OpenGL renderers loading shaders
       void loadGeometry();
       bool isExit();
       bool isCompute();
       void dbgSeed();
  private: 
       void prepareVizScene(); 
  public:
       void uploadGeometryViz();
  public:
       void loadGenstep();
       void targetViz();
       void loadEvtFromFile();
       void uploadEvtViz();

#ifdef WITH_OPTIX
  public:
       void prepareOptiX();
       void setupEventInEngine();
       void preparePropagator();
       void seedPhotonsFromGensteps();
       void initRecords();
       void propagate();
       void saveEvt();
       void indexSequence();
#endif

  public:
       void indexEvt();
       void indexPresentationPrep();
       void indexEvtOld();
  public:
       void prepareGUI();
       void renderLoop();
       void cleanup();

  public:
       bool hasOpt(const char* name);

  private:
       Opticks*         m_opticks ; 
       OpticksHub*      m_hub ; 
       OpticksIdx*      m_idx ; 
#ifdef WITH_OPTIX
       OpEngine*        m_ope ; 
       OpViz*           m_opv ; 
#endif
       OpticksViz*      m_viz ; 

};

#include "GGV_TAIL.hh"


