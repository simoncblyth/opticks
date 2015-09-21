#pragma once
#include <cstring>
#include <glm/glm.hpp>

template<typename T> class NPY ; 

class TorchStepNPY {
   public:
       typedef enum { POS_TARGET, DIR_TARGET, NUM_PHOTONS, MATERIAL_LINE, DIRECTION, ZENITH_AZIMUTH, WAVELENGTH, WEIGHT, TIME, UNRECOGNIZED } Param_t ;

       static const char* DEFAULT_CONFIG ; 

       static const char* POS_TARGET_ ; 
       static const char* DIR_TARGET_ ; 
       static const char* NUM_PHOTONS_ ; 
       static const char* MATERIAL_LINE_ ; 
       static const char* DIRECTION_ ; 
       static const char* ZENITH_AZIMUTH_ ; 
       static const char* WAVELENGTH_ ; 
       static const char* WEIGHT_ ; 
       static const char* TIME_ ; 
   public:  
       TorchStepNPY(unsigned int genstep_id, const char* config=NULL); 
       void configure(const char* config);
       NPY<float>* makeNPY();
   private:
       Param_t getParam(const char* k);
       void set(TorchStepNPY::Param_t param, const char* s );
   public:  
       // target setting needs external info regarding geometry 
       void setPosTarget(const char* s );
       void setDirTarget(const char* s );
       glm::ivec4&  getPosTarget();
       glm::ivec4&  getDirTarget();
       void setPosition(glm::vec3& pos);
       void setDirection(glm::vec3& dir);
   public:  
       // currently ignored on the GPU
       void setPolarization(glm::vec3& pol);
   public:  
       void setNumPhotons(const char* s );
       void setMaterialLine(const char* s );
       void setDirection(const char* s );
       void setZenithAzimuth(const char* s );
       void setWavelength(const char* s );
       void setWeight(const char* s );
       void setTime(const char* s );

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
       void setGenstepId(int id);  /* MANDATORY TO SET THIS TO TORCH */
  private:
       const char*  m_config ;
  private:
       glm::ivec4   m_pos_target ;
       glm::ivec4   m_dir_target ;
  private:
       // 6 quads that are copied into the genstep 
       glm::ivec4   m_ctrl ;
       glm::vec4    m_post ;
       glm::vec4    m_dirw ;
       glm::vec4    m_polw ;
       glm::vec4    m_zenith_azimuth ;
       glm::vec4    m_spare ; 
  private:
       NPY<float>*  m_npy ; 
 
};



inline TorchStepNPY::TorchStepNPY(unsigned int id, const char* config) 
       :  
       m_config(config ? strdup(config) : DEFAULT_CONFIG),
       m_npy(NULL)
{
   setGenstepId(id);
   configure(m_config);
}


inline glm::ivec4& TorchStepNPY::getPosTarget()
{
    return m_pos_target ; 
}

inline glm::ivec4& TorchStepNPY::getDirTarget()
{
    return m_dir_target ; 
}



inline void TorchStepNPY::setGenstepId(int id)
{
   m_ctrl.x = id ; 
}
inline void TorchStepNPY::setPosition(glm::vec3& pos)
{
    m_post.x = pos.x ; 
    m_post.y = pos.y ; 
    m_post.z = pos.z ; 
}
inline void TorchStepNPY::setDirection(glm::vec3& dir)
{
    m_dirw.x = dir.x ; 
    m_dirw.y = dir.y ; 
    m_dirw.z = dir.z ; 
}
inline void TorchStepNPY::setPolarization(glm::vec3& pol)
{
    m_polw.x = pol.x ; 
    m_polw.y = pol.y ; 
    m_polw.z = pol.z ; 
}



