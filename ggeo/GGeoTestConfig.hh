#pragma once

#include <string>
#include <vector>
#include <map>

#include <glm/fwd.hpp>
struct NSlice ; 

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"


/**

GGeoTestConfig
===============

Parses a configure string into the specifications of 
simple test geometries.   The specification is used
by both Geant4 and Opticks to create corresponding geometries.
The Geant4 usage is done via :doc:`../cfg4/CTestDetector`.


**/

class GGEO_API GGeoTestConfig {
    public:
       typedef enum { 
                      MODE, 
                      FRAME, 
                      BOUNDARY, 
                      PARAMETERS, 
                      SHAPE, 
                      SLICE, 
                      ANALYTIC, 
                      DEBUG,
                      CONTROL,
                      PMTPATH,
                      UNRECOGNIZED } Arg_t ;

       typedef std::pair<std::string,std::string> KV ; 
       static const char* DEFAULT_CONFIG ; 
    public:
       static const char* MODE_; 
       static const char* FRAME_ ; 
       static const char* BOUNDARY_ ; 
       static const char* PARAMETERS_ ; 
       static const char* SHAPE_ ; 
       static const char* SLICE_ ; 
       static const char* ANALYTIC_ ; 
       static const char* DEBUG_ ; 
       static const char* CONTROL_ ; 
       static const char* PMTPATH_ ; 
    public:
       GGeoTestConfig(const char* config);
       int getVerbosity();
    private:
       void init(const char* config);
       void configure(const char* config);
       Arg_t getArg(const char* k);
       void set(Arg_t arg, const char* s);
    private:
       void setMode(const char* s);
       void setFrame(const char* s);
       void setSlice(const char* s);
       void setAnalytic(const char* s);
       void setDebug(const char* s);
       void setControl(const char* s);
       void setPmtPath(const char* s);
    private:
       void addShape(const char* s);
       void addBoundary(const char* s);
       void addParameters(const char* s);
    public:
       const char* getBoundary(unsigned int i);
       glm::vec4 getParameters(unsigned int i);
       char      getShape(unsigned int i);
       std::string getShapeString(unsigned int i); 

       NSlice*   getSlice();
       bool      getAnalytic();
       const char* getMode();
       const char* getPmtPath();
       unsigned int getNumElements();

       std::vector<std::pair<std::string, std::string> >& getCfg();
       void dump(const char* msg="GGeoTestConfig::dump");
   private:
       unsigned int getNumBoundaries();
       unsigned int getNumParameters();
       unsigned int getNumShapes();
   private:
       const char*  m_config ; 
       const char*  m_mode ; 
       const char*  m_pmtpath ; 
       NSlice*      m_slice ; 
       glm::ivec4   m_frame ;
       glm::ivec4   m_analytic ;
       glm::vec4    m_debug ;
       glm::ivec4   m_control ;
       std::vector<std::string> m_shapes ; 
       std::vector<std::string> m_boundaries ; 
       std::vector<glm::vec4>   m_parameters ; 
       std::vector<KV> m_cfg ; 
};

#include "GGEO_TAIL.hh"


