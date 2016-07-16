#pragma once

// for both non-CUDA and CUDA compilation
typedef enum {
   T_UNDEF,
   T_SPHERE,
   T_POINT,
   T_DISC,
   T_DISC_INTERSECT_SPHERE,
   T_DISC_INTERSECT_SPHERE_DUMB,
   T_DISCLIN,
   T_DISCAXIAL,
   T_INVSPHERE,
   T_REFLTEST,
   T_INVCYLINDER,
   T_RING, 
   T_NUM_TYPE
}               Torch_t ;

typedef enum {
   M_UNDEF             = 0x0 ,
   M_SPOL              = 0x1 << 0,
   M_PPOL              = 0x1 << 1,
   M_FLAT_THETA        = 0x1 << 2, 
   M_FLAT_COSTHETA     = 0x1 << 3,
   M_FIXPOL            = 0x1 << 4,
   M_WAVELENGTH_SOURCE = 0x1 << 5,
   M_WAVELENGTH_COMB   = 0x1 << 6
}              Mode_t ; 


#ifndef __CUDACC__


#include <string>
#include "NGLM.hpp"

template<typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API TorchStepNPY {
   public:
       typedef enum { TYPE, 
                      MODE, 
                      POLARIZATION, 
                      FRAME,  
                      TRANSFORM, 
                      SOURCE, 
                      TARGET, 
                      PHOTONS, 
                      MATERIAL, 
                      ZENITHAZIMUTH, 
                      WAVELENGTH, 
                      WEIGHT, 
                      TIME, 
                      RADIUS, 
                      DISTANCE, 
                      UNRECOGNIZED } Param_t ;

       static const char* DEFAULT_CONFIG ; 

       static const char* TYPE_; 
       static const char* MODE_; 
       static const char* POLARIZATION_; 
       static const char* FRAME_ ; 
       static const char* TRANSFORM_ ; 
       static const char* SOURCE_ ; 
       static const char* TARGET_ ; 
       static const char* PHOTONS_ ; 
       static const char* MATERIAL_ ; 
       static const char* ZENITHAZIMUTH_ ; 
       static const char* WAVELENGTH_ ; 
       static const char* WEIGHT_ ; 
       static const char* TIME_ ; 
       static const char* RADIUS_ ; 
       static const char* DISTANCE_ ; 

       static const char* T_UNDEF_ ; 
       static const char* T_SPHERE_ ; 
       static const char* T_POINT_ ; 
       static const char* T_DISC_ ; 
       static const char* T_DISCLIN_ ; 
       static const char* T_DISCAXIAL_ ; 
       static const char* T_DISC_INTERSECT_SPHERE_ ; 
       static const char* T_DISC_INTERSECT_SPHERE_DUMB_ ; 
       static const char* T_INVSPHERE_ ; 
       static const char* T_REFLTEST_ ; 
       static const char* T_INVCYLINDER_ ; 
       static const char* T_RING_ ; 

       static const char* M_SPOL_ ; 
       static const char* M_PPOL_ ; 
       static const char* M_FLAT_THETA_ ; 
       static const char* M_FLAT_COSTHETA_ ; 
       static const char* M_FIXPOL_ ; 
       static const char* M_WAVELENGTH_SOURCE_ ; 
       static const char* M_WAVELENGTH_COMB_ ; 

   public:  
       TorchStepNPY(unsigned int genstep_id, unsigned int num_step=1, const char* config=NULL); 
       void configure(const char* config);
       void addStep(bool verbose=false); // increments m_step_index
       NPY<float>* getNPY();
       void update();
   private:
       ::Mode_t  parseMode(const char* k);
       ::Torch_t parseType(const char* k);
       Param_t parseParam(const char* k);
       void set(TorchStepNPY::Param_t param, const char* s );
   public:  
       void setMode(const char* s );
       void setType(const char* s );
   public:  
       // target setting needs external info regarding geometry 
       void setFrame(const char* s );
       void setFrame(unsigned int vindex );
       glm::ivec4&  getFrame();
       void setFrameTransform(glm::mat4& transform);
       // targetting needs frame transform info which is done by GGeo::targetTorchStep(torchstep)

       void setFrameTransform(const char* s );       // directly from string of 16 comma delimited floats 
       void setFrameTargetted(bool targetted=true);
       bool isFrameTargetted();
       const glm::mat4& getFrameTransform();
   public:
       // slots used by Geant4 only (not Opticks) from cfg4- 
       void setNumPhotonsPerG4Event(unsigned int n);
       unsigned int getNumPhotonsPerG4Event(); 
       unsigned int getNumG4Event();
       bool isIncidentSphere();
       bool isDisc();
       bool isDiscLinear();
       bool isRing();
       bool isPoint();
       bool isReflTest();
       bool isSPolarized();
       bool isPPolarized();
       bool isFixPolarized();
       void Summary(const char* msg="TorchStepNPY::Summary");
   public:
       // local positions/vectors, frame transform is applied in *update* yielding world frame m_post m_dirw 
       void setSourceLocal(const char* s );
       void setTargetLocal(const char* s );
       void setPolarizationLocal(const char* s );
       glm::vec4& getSourceLocal();
       glm::vec4& getTargetLocal();
       glm::vec4& getPolarizationLocal();
   public:  
       // need external help to set the MaterialLine
       void setMaterial(const char* s );
       const char* getConfig();
       const char* getMaterial();
   public:  
        // methods invoked by update after frame transform is available
       void setPosition(const glm::vec4& pos );
       void setDirection(const glm::vec3& dir );
       void setPolarization(const glm::vec4& pol );
   public:  
       void setNumPhotons(const char* s );
       void setMaterialLine(unsigned int ml);
       void setDirection(const char* s );
       void setZenithAzimuth(const char* s );
       void setWavelength(const char* s );
       void setWeight(const char* s );
       void setTime(const char* s );
       void setRadius(const char* s );
       void setDistance(const char* s );

       void setNumPhotons(unsigned int num_photons );
       void setRadius(float radius );
       void setDistance(float distance);
   public:  
       glm::vec3 getPosition();
       glm::vec3 getDirection();
       glm::vec3 getPolarization();
       glm::vec4 getZenithAzimuth();

       float getTime();
       float getRadius();
       float getWavelength();


   public:  
       ::Mode_t  getMode();
       ::Torch_t getType();
       std::string getModeString();
       const char* getTypeName();
       unsigned int getNumPhotons();
       unsigned int getMaterialLine();


/*
   *setZenithAzimuth*

   Photons directions are generated using two random numbers in range 0:1 
   which are used scale the zenith and azimuth ranges.
   Default is a uniform sphere. Changing zenith ranges allows cones or
   rings to be generated and changing azimuth range allows 
   to chop the cone, ring or sphere.

                       mapped to 0:2pi of azimuth angle    
                    -------
           (0.f,1.f,0.f,1.f)
            --------
              mapped to 0:pi of zenith angle
*/

       void dump(const char* msg="TorchStepNPY::dump");
  private:
       void setGenstepId(); 
  private:
       unsigned int m_genstep_id ; 
       const char*  m_config ;
       const char*  m_material ;
  private:
       glm::ivec4   m_frame ;
       glm::mat4    m_frame_transform ; 
       bool         m_frame_targetted ; 
  private:
       // position and directions to which the frame transform is applied in update
       glm::vec4    m_source_local ; 
       glm::vec4    m_target_local ; 
       glm::vec4    m_polarization_local ; 
  private:
       glm::vec4    m_src ;
       glm::vec4    m_tgt ;
       glm::vec4    m_pol ;
       glm::vec3    m_dir ;
  private:
       // 6 quads that are copied into the genstep and passed to GPU cu/torchstep.h
       glm::ivec4   m_ctrl ;
       glm::vec4    m_post ;
       glm::vec4    m_dirw ;
       glm::vec4    m_polw ;
       glm::vec4    m_zeaz ;
       glm::vec4    m_beam ; 
  private:
       unsigned int m_num_step ; 
       unsigned int m_step_index ; 
       NPY<float>*  m_npy ; 
  private:
       unsigned int m_num_photons_per_g4event ;
 
};

#include "NPY_TAIL.hh"


#endif


