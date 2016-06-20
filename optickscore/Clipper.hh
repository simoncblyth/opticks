#pragma once 


#include <string>
#include <vector>

#include <glm/fwd.hpp>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

#include "NConfigurable.hpp"

class OKCORE_API Clipper : public NConfigurable {
   public:
        static const char* PREFIX ;
        const char* getPrefix();
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

   public:
        void gui(); 

   private:
        void setAbsolute(bool absolute);
        void update(glm::mat4& model_to_world);

   public:
        float* getPointPtr();
        float* getNormalPtr();
        float* getPlanePtr();

   private:
        int       m_mode ; 
        bool      m_absolute ; 

   private:
        // input model frame definition of clipping plane
        glm::vec3 m_point  ; 
        glm::vec3 m_normal ; 
   
   private:
        // derived world frame qtys
        glm::vec3 m_wpoint ; 
        glm::vec3 m_wnormal ;
        glm::vec4 m_wplane ;   // plane derived from point, normal and model_to_world

   private:
        // pig in poke
        glm::vec4 m_absplane ;  // plane directly set in world frame

        float*    m_float3 ; 
};

#include "OKCORE_TAIL.hh"

