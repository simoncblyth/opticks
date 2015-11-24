#pragma once

// for both non-CUDA and CUDA compilation
typedef enum {
   T_UNDEF,
   T_SPHERE,
   T_DISC,
   T_DISCLIN,
   T_DISCAXIAL,
   T_INVSPHERE,
   T_REFLTEST,
   T_INVCYLINDER,
   T_NUM_TYPE
}               Torch_t ;

typedef enum {
   M_UNDEF,
   M_SPOL,
   M_PPOL,
   M_NUM_POLZ
}              Polz_t ; 


#ifndef __CUDACC__

#include <cstring>
#include <glm/glm.hpp>

template<typename T> class NPY ; 

class TorchStepNPY {
   public:
       typedef enum { TYPE, POLZ, FRAME, TRANSFORM, SOURCE, TARGET, PHOTONS, MATERIAL, ZENITHAZIMUTH, WAVELENGTH, WEIGHT, TIME, RADIUS, DISTANCE, UNRECOGNIZED } Param_t ;

       static const char* DEFAULT_CONFIG ; 

       static const char* TYPE_; 
       static const char* POLZ_; 
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

       static const char* T_SPHERE_ ; 
       static const char* T_DISC_ ; 
       static const char* T_DISCLIN_ ; 
       static const char* T_DISCAXIAL_ ; 
       static const char* T_INVSPHERE_ ; 
       static const char* T_REFLTEST_ ; 
       static const char* T_INVCYLINDER_ ; 

       static const char* M_SPOL_ ; 
       static const char* M_PPOL_ ; 

   public:  
       TorchStepNPY(unsigned int genstep_id, unsigned int num_step=1, const char* config=NULL); 
       void configure(const char* config);
       void addStep(bool verbose=false); // increments m_step_index
       NPY<float>* getNPY();
   private:
       void update();
       ::Polz_t  parsePolz(const char* k);
       ::Torch_t parseType(const char* k);
       Param_t parseParam(const char* k);
       void set(TorchStepNPY::Param_t param, const char* s );
   public:  
       void setPolz(const char* s );
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
       // local positions, frame transform is applied in *update* yielding world frame m_post m_dirw 
       void setSourceLocal(const char* s );
       void setTargetLocal(const char* s );
       glm::vec4& getSourceLocal();
       glm::vec4& getTargetLocal();
   public:  
       // currently ignored on the GPU
       void setPolarization(glm::vec3& pol);
   public:  
       // need external help to set the MaterialLine
       void setMaterial(const char* s );
       const char* getConfig();
       const char* getMaterial();
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


       ::Polz_t  getPolz();
       ::Torch_t getType();
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
       glm::vec4    m_source_local ; 
       glm::vec4    m_target_local ; 
       glm::vec4    m_src ;
       glm::vec4    m_tgt ;
       glm::vec3    m_dir ;
  private:
       // 6 quads that are copied into the genstep 
       glm::ivec4   m_ctrl ;
       glm::vec4    m_post ;
       glm::vec4    m_dirw ;
       glm::vec4    m_polw ;
       glm::vec4    m_zenith_azimuth ;
       glm::vec4    m_beam ; 
  private:
       unsigned int m_num_step ; 
       unsigned int m_step_index ; 
       NPY<float>*  m_npy ; 
 
};



inline TorchStepNPY::TorchStepNPY(unsigned int genstep_id, unsigned int num_step, const char* config) 
       :  
       m_genstep_id(genstep_id), 
       m_config(config ? strdup(config) : DEFAULT_CONFIG),
       m_material(NULL),
       m_frame_targetted(false),
       m_num_step(num_step),
       m_step_index(0),
       m_npy(NULL)
{
   configure(m_config);
}


inline glm::ivec4& TorchStepNPY::getFrame()
{
    return m_frame ; 
}
inline void TorchStepNPY::setFrameTransform(glm::mat4& frame_transform)
{
    m_frame_transform = frame_transform ;
}
inline const glm::mat4& TorchStepNPY::getFrameTransform()
{
    return m_frame_transform ;
}

inline void TorchStepNPY::setFrameTargetted(bool targetted)
{
    m_frame_targetted = targetted ;
}
inline bool TorchStepNPY::isFrameTargetted()
{
    return m_frame_targetted ;
} 



inline glm::vec4& TorchStepNPY::getSourceLocal()
{
    return m_source_local ; 
}
inline glm::vec4& TorchStepNPY::getTargetLocal()
{
    return m_target_local ; 
}
inline void TorchStepNPY::setGenstepId()
{
   m_ctrl.x = m_genstep_id ; 
}
inline void TorchStepNPY::setPolarization(glm::vec3& pol)
{
    m_polw.x = pol.x ; 
    m_polw.y = pol.y ; 
    m_polw.z = pol.z ; 
}
inline const char* TorchStepNPY::getMaterial()
{
    return m_material ; 
}
inline const char* TorchStepNPY::getConfig()
{
    return m_config ; 
}

#endif


