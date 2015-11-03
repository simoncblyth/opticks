#pragma once
#include <cstring>
#include <glm/glm.hpp>

struct gbbox ; 
class GGeo ; 
class GMesh ; 

class GTestBox {
   public:
       enum { NUM_VERTICES = 24, NUM_FACES = 6*2 } ;

       typedef enum { FRAME, 
                      BOUNDARY, 
                      UNRECOGNIZED } Param_t ;

       static const char* DEFAULT_CONFIG ; 
   public:
       static const char* FRAME_ ; 
       static const char* BOUNDARY_ ; 
   public:
       GTestBox(GGeo* ggeo, const char* config=NULL);
       void make();
   private:
       void configure(const char* config);
       Param_t getParam(const char* k);
       void set(Param_t p, const char* s);
   public:
       void setFrame(const char* s);
       void setBoundary(const char* s);
   public:
       GMesh* makeMesh(unsigned int index, gbbox& bb); 
   public:
       void dump(const char* msg="GTestBox::dump");
   private:
       GGeo*        m_ggeo ;  
       GMesh*       m_mesh ; 
       const char*  m_config ; 
       glm::ivec4   m_frame ;
       unsigned int m_boundary ; 

};


inline GTestBox::GTestBox(GGeo* ggeo, const char* config)
    :
    m_ggeo(ggeo),
    m_mesh(NULL)
{
    configure(config);
}

