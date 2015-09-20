#pragma once
#include <cstring>
#include <glm/glm.hpp>

template<typename T> class NPY ; 

class TorchStepNPY {
   public:  
       TorchStepNPY(const char* config); 
       unsigned int getTarget();

       void setCtrl(int Id, int ParentId, int MaterialLine, int NumPhotons);
       void setPositionTime(glm::vec3& pos, float time=0.f);
       void setDirectionWeight(glm::vec3& dir, float weight=1.f);
       void setPolarizationWavelength(glm::vec3& pol, float wavelength=500.f);

       NPY<float>* makeNPY();
  private:
       void parseConfig();
  private:
       const char*  m_config ;
       unsigned int m_target ;  
       glm::ivec4   m_ctrl ;
       glm::vec4    m_post ;
       glm::vec4    m_dirw ;
       glm::vec4    m_polw ;
  private:
       NPY<float>*  m_npy ; 
 
};



inline TorchStepNPY::TorchStepNPY(const char* config) 
       :  
       m_config(strdup(config)),
       m_target(0),
       m_npy(NULL)
{
   parseConfig();
}

inline unsigned int TorchStepNPY::getTarget()
{
    return m_target ; 
}

inline void TorchStepNPY::setCtrl(int Id, int ParentId, int MaterialLine, int NumPhotons)
{
   m_ctrl.x = Id ; 
   m_ctrl.y = ParentId ; 
   m_ctrl.z = MaterialLine ; 
   m_ctrl.w = NumPhotons ; 
}

inline void TorchStepNPY::setPositionTime(glm::vec3& pos, float time)
{
    m_post.x = pos.x ; 
    m_post.y = pos.y ; 
    m_post.z = pos.z ; 
    m_post.w = time ; 
}

inline void TorchStepNPY::setDirectionWeight(glm::vec3& dir, float weight)
{
    m_dirw.x = dir.x ; 
    m_dirw.y = dir.y ; 
    m_dirw.z = dir.z ; 
    m_dirw.w = weight ; 
}

inline void TorchStepNPY::setPolarizationWavelength(glm::vec3& pol, float wavelength)
{
    m_polw.x = pol.x ; 
    m_polw.y = pol.y ; 
    m_polw.z = pol.z ; 
    m_polw.w = wavelength ; 
}




