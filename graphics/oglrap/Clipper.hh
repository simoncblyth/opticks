#pragma once 

#include <glm/glm.hpp>  
#include <string>
#include <vector>


// how to handle multiple planes ?
class Clipper {
   public:
        static const char* CUTPOINT ;
        static const char* CUTNORMAL ;
        static const char* CUTMODE  ;

        Clipper();
        glm::vec4& getClipPlane(glm::mat4& model_to_world);
        void dump(const char* msg);

        void set(const char* name, std::string& _xyz);
        void configureS(const char* name, std::vector<std::string> values);
        void configureI(const char* name, std::vector<int> values);

        void setPoint( glm::vec3& point);
        void setNormal(glm::vec3& normal);
        void setMode(int mode);

        int getMode();
        void next();

   private:
        void update(glm::mat4& model_to_world);

   private:
        int       m_mode ; 
        glm::vec3 m_point  ; 
        glm::vec3 m_normal ; 
   
        glm::vec3 m_wpoint ; 
        glm::vec3 m_wnormal ;
        glm::vec4 m_wplane ;

};



inline void Clipper::setMode(int mode)
{
    m_mode = mode ;
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


