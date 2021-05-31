#pragma once

#include <string>

#include "plog/Severity.h"
#include "NGLM.hpp"

template<typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

/**
NStep
======

This functionality was formerly in GenstepNPY 
but it is useful to separate out the handling of a 
single step for when the complexities of multi-steps
and frame transforms are not needed. 

CAUTION : MEANINGS OF GENSTEP ELEMENTS VARY WITH THE TYPE OF GENSTEP

**/

struct NPY_API NStep
{
    static const plog::Severity LEVEL ; 

    NPY<float>*  m_array ; 
    bool         m_filled ; 

    // 6 transport quads that are copied into the genstep buffer by addStep
    glm::ivec4   m_ctrl ;
    glm::vec4    m_post ;
    glm::vec4    m_dirw ;
    glm::vec4    m_polw ;
    glm::vec4    m_zeaz ;
    glm::vec4    m_beam ; 

   public:  
       NStep();  

   public:  
       std::string desc(const char* msg="NStep::desc") const ; 

   public:
        void        fillArray() ; 
        NPY<float>* getArray() const ;

   public:  
       // m_ctrl
       void setGenstepType(unsigned gentype);  
       void setOriginTrackID(unsigned trackID);
       void setMaterialLine(unsigned int ml);
       void setNumPhotons(const char* s );
       void setNumPhotons(unsigned int num_photons );

       unsigned getGenstepType() const ;
       unsigned getOriginTrackID() const ; 
       unsigned getMaterialLine() const ;
       unsigned getNumPhotons() const ;

   public:  
       // m_post
    
       void setPosition(const glm::vec4& pos );
       void setDirection(const glm::vec3& dir );
       void setPolarization(const glm::vec4& pol );

       void setDirection(const char* s );
       void setZenithAzimuth(const char* s );
       void setWavelength(const char* s );
       void setWeight(const char* s );
       void setTime(const char* s );
       void setRadius(const char* s );
       void setDistance(const char* s );

       void setRadius(float radius );
       void setDistance(float distance);
       void setBaseMode(unsigned umode);
       void setBaseType(unsigned utype);
   public:  
       glm::vec3 getPosition() const ;
       glm::vec3 getDirection() const ;
       glm::vec3 getPolarization() const ;
       glm::vec4 getZenithAzimuth() const ;

       float getTime() const ;
       float getRadius() const ;
       float getWavelength() const ;


       unsigned getBaseMode() const ;
       unsigned getBaseType() const ;



};


#include "NPY_TAIL.hh"

