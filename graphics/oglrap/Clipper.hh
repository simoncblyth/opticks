#pragma once 

#include <glm/glm.hpp>  
#include <string>
#include <vector>


// how to handle multiple planes ?

#include "Configurable.hh"

class Clipper : public Configurable {
   public:
        static const char* CUTPOINT ;
        static const char* CUTNORMAL ;
        static const char* CUTMODE  ;
        static const char* CUTPLANE ;
        static const char* CUTPRINT ;

        Clipper();

 public:
        // Configurable
        static bool accepts(const char* name);
        void configure(const char* name, const char* value);
        std::vector<std::string> getTags();
        void set(const char* name, std::string& xyz);
        std::string get(const char* name);


        glm::vec4& getClipPlane(glm::mat4& model_to_world);
        void dump(const char* msg);

        //void set(const char* name, std::string& _xyz);
        void configureS(const char* name, std::vector<std::string> values);
        void configureI(const char* name, std::vector<int> values);

        void setMode(int mode);
        int getMode();
        void next();  // toggles mode, Interactor invokes this on pressing C


        // TODO:  try to eliminate the modal switch by some input side helpers 
        //        to make it easy to enter abs planes from non-abs inputs perhaps?

   public:
        // using setPoint or setNormal switches absolute mode OFF
        void setPoint( glm::vec3& point);
        void setNormal(glm::vec3& normal);
        glm::vec3& getPoint();
        glm::vec3& getNormal();

   public:
        // using setPlane switches absolute mode ON
        void setPlane( glm::vec4& plane);
        glm::vec4& getPlane();

   private:
        void setAbsolute(bool absolute);
        void update(glm::mat4& model_to_world);

   private:
        int       m_mode ; 
        bool      m_absolute ; 
        glm::vec3 m_point  ; 
        glm::vec3 m_normal ; 
   
        glm::vec3 m_wpoint ; 
        glm::vec3 m_wnormal ;
        glm::vec4 m_wplane ;   // plane derived from point, normal and model_to_world

        glm::vec4 m_absplane ;  // plane directly set in world frame

};




inline glm::vec3& Clipper::getPoint()
{
    return m_point ; 
}
inline glm::vec3& Clipper::getNormal()
{
    return m_normal ; 
}
inline glm::vec4& Clipper::getPlane()
{
    return m_absplane ; 
}


inline int Clipper::getMode()
{
    return m_mode ; 
}

inline void Clipper::next()
{
    // Interactor invokes this on pressing C, for now just toggle between -1 and 0
    m_mode = m_mode != -1 ? -1 : 0 ; 
}


