#include "Clipper.hh"

// npy-
#include "GLMPrint.hpp"

#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

Clipper::Clipper() :
   m_mode(-1),
   m_point(0,0,0),
   m_normal(1,0,0)
{
}


const char* Clipper::CUTMODE    = "cutmode" ;
const char* Clipper::CUTPOINT   = "cutpoint" ;
const char* Clipper::CUTNORMAL  = "cutnormal" ;

void Clipper::configureI(const char* name, std::vector<int> values)
{
    if(values.empty()) return ;
    int last = values.back();

    if(     strcmp(name,CUTMODE)==0) setMode(last);  
    else
          printf("Clipper::configureI bad name %s\n", name);
}

void Clipper::configureS(const char* name, std::vector<std::string> values)
{
    if(values.empty()) return ;

    std::string last = values.back();
    set(name, last);
}


void Clipper::set(const char* name, std::string& _xyz)
{
    std::vector<std::string> xyz;
    boost::split(xyz, _xyz, boost::is_any_of(","));

    if(xyz.size() == 3 )
    {
        float x = boost::lexical_cast<float>(xyz[0]); 
        float y = boost::lexical_cast<float>(xyz[1]); 
        float z = boost::lexical_cast<float>(xyz[2]); 

        glm::vec3 v(x, y, z);

        if(     strcmp(name,CUTPOINT)==0)   setPoint(v);
        else if(strcmp(name,CUTNORMAL) == 0 ) setNormal(v);
        else
              printf("Clipper::configureS bad name %s\n", name);
    }
    else
    {
        printf("Clipper::set malformed %s : %s \n", name, _xyz.c_str() );
    }
}

void Clipper::setPoint(glm::vec3& point)
{
    m_point = point ;
}

void Clipper::setNormal(glm::vec3& normal)
{
    m_normal = normal ;
}






void Clipper::update(glm::mat4& model_to_world)
{
    // transform point position and normal direction from model to world frame
    //
    // model_to_world does uniform extent scaling and a translation only
    // so does not change directions

    m_wnormal = glm::normalize( m_normal );

    m_wpoint = glm::vec3( model_to_world * glm::vec4(m_point,  1.f));

    m_wplane = glm::vec4( m_wnormal, -glm::dot( m_wpoint, m_wnormal ));  // plane equation for OpenGL 
}


glm::vec4& Clipper::getClipPlane(glm::mat4& model_to_world)
{
    update(model_to_world);

    //print(model_to_world, "Clipper::getClipPlane model_to_world");
    //dump("Clipper::getClipPlane");

    return m_wplane ; 
}

void Clipper::dump(const char* msg)
{ 
    printf("%s\n", msg);

    print( m_normal,  "m_normal : (model frame) vector normal to plane  ");
    print( m_point,   "m_point  : (model frame) point in the plane ");
    print( m_wnormal, "m_wnormal : (world frame) normalized normal to plane ");
    print( m_wpoint,  "m_wpoint  : (world frame) point in the plane  ");
    print( m_wplane,  "m_wplane  : (world frame) plane equation for shader consumption");

}




/*

Extract from daeclipper.py 

Maths 
------

* http://www.songho.ca/math/plane/plane.html

The equation of a plane is defined with a normal vector
(perpendicular to the plane) and a known point on the plane.::

   ax + by + cz + d = 0

The normal direction gives coefficients (a,b,c) and the single point (x1,y1,z1) 
fixes the plane along that direction via::

   d = -np.dot( [a,b,c], [x1,y1,z1] )  


Implement from daetransform
---------------------------------

To obtain those from current view transforms use:

#. gaze direction `look - eye` (0,0,-1)  [in eye frame] 
#. some convenient point, maybe near point (0,0,-near) or look point (0,0,-distance) [in eye frame] 

Actually more convenient to use the current near clipping plain, as 
can interactively vary that and when have it where desired freeze it
into a fixed clipping plane.


*/

