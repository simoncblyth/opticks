#pragma once
#include <cstring>
#include <string>
#include <glm/glm.hpp>
#include <cassert>


class NGunConfig {
   public:
       static const char* DEFAULT_CONFIG ; 

       static const char* COMMENT_; 
       static const char* PARTICLE_; 
       static const char* FRAME_ ; 
       static const char* POSITION_ ; 
       static const char* DIRECTION_ ; 
       static const char* POLARIZATION_ ; 
       static const char* TIME_ ; 
       static const char* ENERGY_ ; 
       static const char* NUMBER_ ; 
       static const char* UNRECOGNIZED_ ; 
   public:
       typedef enum { 
                      COMMENT,
                      PARTICLE, 
                      FRAME,  
                      POSITION, 
                      DIRECTION, 
                      POLARIZATION, 
                      TIME, 
                      ENERGY, 
                      NUMBER, 
                      UNRECOGNIZED
                    } Param_t ;
   private:
       Param_t parseParam(const char* k);
       void set(NGunConfig::Param_t param, const char* s );
       const char* getParam(NGunConfig::Param_t param);
   public:
       NGunConfig();
       void Summary(const char* msg="NGunConfig::Summary");
   public:
       void parse(const char* config=NULL);
       void parse(std::string config);
   private:
       void setComment(const char* s);
       void setParticle(const char* s);
       void setNumber(const char* s);
       void setNumber(unsigned int num);
       void setEnergy(const char* s );
       void setEnergy(float energy);
       void setTime(const char* s );
       void setTime(float time);
       void setFrame(const char* s);
       void setPositionLocal(const char* s );
       void setDirectionLocal(const char* s );
       void setPolarizationLocal(const char* s );
   public: 
       void setFrameTransform(const char* s ); // 16 comma delimited floats 
       void setFrameTransform(glm::mat4& transform);
       const glm::mat4& getFrameTransform();
   private:  
       // methods invoked after setFrameTransform
       void update();
       void setPosition(const glm::vec3& pos );
       void setDirection(const glm::vec3& dir );
       void setPolarization(const glm::vec3& pol );
   public:  
       const char* getComment();
       const char* getParticle();
       unsigned int getNumber();
       float       getTime();
       float       getEnergy();
       int         getFrame();
       glm::vec3   getPosition();
       glm::vec3   getDirection();
       glm::vec3   getPolarization();
   private:
       void init();
   private:
       const char* m_config ;
  private:
       // position and directions to which the frame transform is applied in update
       glm::vec4    m_position_local ; 
       glm::vec4    m_direction_local ; 
       glm::vec4    m_polarization_local ; 
  private:
       const char*  m_comment ;
       const char*  m_particle ;
       unsigned int m_number ;  
       float        m_time  ; 
       float        m_energy  ; 
       glm::ivec4   m_frame ;
       glm::vec3    m_position ; 
       glm::vec3    m_direction ; 
       glm::vec3    m_polarization ; 
   private:
       glm::mat4    m_frame_transform ; 


};

inline NGunConfig::NGunConfig()
    :
     m_config(NULL),
     m_comment(NULL),
     m_particle(NULL),
     m_number(1),
     m_time(0.),
     m_energy(0.)
{
    init();
}


inline void NGunConfig::setTime(float time)
{
    m_time = time ; 
}
inline void NGunConfig::setEnergy(float energy)
{
    m_energy = energy ; 
}
inline void NGunConfig::setNumber(unsigned int number)
{
    m_number = number ; 
}


inline void NGunConfig::setFrameTransform(glm::mat4& frame_transform)
{
    m_frame_transform = frame_transform ;
    update();
}
inline const glm::mat4& NGunConfig::getFrameTransform()
{
    return m_frame_transform ;
}

inline void NGunConfig::setPosition(const glm::vec3& pos )
{
    m_position = pos ; 
}
inline void NGunConfig::setDirection(const glm::vec3& dir )
{
    m_direction = dir ; 
}
inline void NGunConfig::setPolarization(const glm::vec3& pol )
{
    m_polarization = pol ; 
}


inline const char* NGunConfig::getComment()
{
    return m_comment ; 
}
inline const char* NGunConfig::getParticle()
{
    return m_particle ; 
}

inline float NGunConfig::getTime()
{
    return m_time ; 
}
inline float NGunConfig::getEnergy()
{
    return m_energy ; 
}

inline int NGunConfig::getFrame()
{
    return m_frame.x ; 
}
inline unsigned int NGunConfig::getNumber()
{
    return m_number ; 
}



inline glm::vec3 NGunConfig::getPosition()
{
    return m_position ; 
}
inline glm::vec3 NGunConfig::getDirection()
{
    return m_direction ; 
}
inline glm::vec3 NGunConfig::getPolarization()
{
    return m_polarization ; 
}




