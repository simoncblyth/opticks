#pragma once

#include <string>
#include <vector>
#include <map>

#include <glm/fwd.hpp>

#include "OpticksCSG.h"

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
      // NODE is a generalization of the former SHAPE argument
       typedef enum { 
                      MODE, 
                      FRAME, 
                      BOUNDARY, 
                      PARAMETERS, 
                      NODE, 
                      ANALYTIC, 
                      DEBUG,
                      CONTROL,
                      PMTPATH,
                      TRANSFORM, 
                      CSGPATH,
                      OFFSETS,
                      NAME,
                      UNRECOGNIZED } Arg_t ;

       typedef std::pair<std::string,std::string> KV ; 
       static const char* DEFAULT_CONFIG ; 
    public:
       static const char* MODE_; 
       static const char* FRAME_ ; 
       static const char* BOUNDARY_ ; 
       static const char* PARAMETERS_ ; 
       static const char* NODE_ ; 
       static const char* ANALYTIC_ ; 
       static const char* DEBUG_ ; 
       static const char* CONTROL_ ; 
       static const char* PMTPATH_ ; 
       static const char* TRANSFORM_ ; 
       static const char* CSGPATH_ ;   // not yet used
       static const char* OFFSETS_ ; 
       static const char* NAME_ ; 
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
       void setAnalytic(const char* s);
       void setDebug(const char* s);
       void setControl(const char* s);
       void setPmtPath(const char* s);
       void setCsgPath(const char* s);
       void setOffsets(const char* s);
       void setName(const char* s);
    private:
       void addNode(const char* s);
       void addBoundary(const char* s);
       void addParameters(const char* s);
       void addTransform(const char* s);
    public:
       const char* getBoundary(unsigned int i);
       glm::vec4 getParameters(unsigned int i);
       glm::mat4 getTransform(unsigned int i);
       //char      getNode(unsigned int i);  use of char codes, a workaround for dependency issue pre enum-unification ?
       OpticksCSG_t getTypeCode(unsigned int i);
       std::string getNodeString(unsigned int i); 

       bool      getAnalytic();
       const char* getMode();
       const char* getPmtPath();
       const char* getCsgPath();
       const char* getName();
       unsigned int getNumElements();

       std::vector<std::pair<std::string, std::string> >& getCfg();
       void dump(const char* msg="GGeoTestConfig::dump");
   private:
       unsigned getNumBoundaries();
       unsigned getNumParameters();
       unsigned getNumNodes();
       unsigned getNumTransforms();
   private:
       unsigned getOffset(unsigned idx);
   public: 
       unsigned getNumOffsets();
       bool isStartOfOptiXPrimitive(unsigned nodeIdx );
   private:
       const char*  m_config ; 
       const char*  m_mode ; 
       const char*  m_pmtpath ; 
       const char*  m_csgpath ; 
       const char*  m_name ; 
       glm::ivec4   m_frame ;
       glm::ivec4   m_analytic ;
       glm::vec4    m_debug ;
       glm::ivec4   m_control ;
       std::vector<std::string> m_nodes ; 
       std::vector<unsigned>    m_offsets ;  // identifies which nodes belong to which primitive via node offset indices 
       std::vector<std::string> m_boundaries ; 
       std::vector<glm::vec4>   m_parameters ; 
       std::vector<glm::mat4>   m_transforms ; 
       std::vector<KV> m_cfg ; 
};

#include "GGEO_TAIL.hh"


