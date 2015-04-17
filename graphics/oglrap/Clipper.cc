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
   m_normal(1,0,0),
   m_absolute(false), 
   m_absplane(1,0,0,1)   // placeholder 
{
}


const char* Clipper::CUTMODE    = "cutmode" ;
const char* Clipper::CUTPOINT   = "cutpoint" ;
const char* Clipper::CUTNORMAL  = "cutnormal" ;
const char* Clipper::CUTPLANE   = "cutplane" ;
const char* Clipper::CUTPRINT   = "cutprint" ;

void Clipper::configureI(const char* name, std::vector<int> values)
{
    if(values.empty()) return ;
    int last = values.back();

    if(          strcmp(name,CUTMODE)==0)  setMode(last);  
    else if(     strcmp(name,CUTPRINT)==0) dump("Clipper::configureI");  
    else
          printf("Clipper::configureI bad name %s\n", name);
}

void Clipper::configureS(const char* name, std::vector<std::string> values)
{
    if(values.empty()) return ;

    std::string last = values.back();
    set(name, last);
}


void Clipper::set(const char* name, std::string& arg_)
{
    std::vector<std::string> arg;
    boost::split(arg, arg_, boost::is_any_of(","));

    if(arg.size() == 3 )
    {
        float x = boost::lexical_cast<float>(arg[0]); 
        float y = boost::lexical_cast<float>(arg[1]); 
        float z = boost::lexical_cast<float>(arg[2]); 

        glm::vec3 v(x, y, z);

        if(     strcmp(name,CUTPOINT)==0)     setPoint(v);
        else if(strcmp(name,CUTNORMAL) == 0 ) setNormal(v);
        else
              printf("Clipper::configureS bad name %s\n", name);
    }
    else if(arg.size() == 4)
    {
        float x = boost::lexical_cast<float>(arg[0]); 
        float y = boost::lexical_cast<float>(arg[1]); 
        float z = boost::lexical_cast<float>(arg[2]); 
        float w = boost::lexical_cast<float>(arg[3]); 

        glm::vec4 v(x, y, z, w);

        if(     strcmp(name,CUTPLANE)==0)   setPlane(v);
        else
              printf("Clipper::configureS bad name %s\n", name);

    }
    else
    {
        printf("Clipper::set malformed %s : %s \n", name, arg_.c_str() );
    }
}

void Clipper::setMode(int mode)
{
    m_mode = mode ;
}
void Clipper::setAbsolute(bool absolute)
{
    LOG(info)<<"Clipper::setAbsolute m_absolute -> absolute " << m_absolute << " -> " << absolute ; 
    m_absolute = absolute ;
}

void Clipper::setPoint(glm::vec3& point)
{
    LOG(info)<<"Clipper::setPoint m_abolute: " << m_absolute ; 
    m_point = point ;
    setAbsolute(false);
}
void Clipper::setNormal(glm::vec3& normal)
{
    LOG(info)<<"Clipper::setNormal m_abolute: " << m_absolute ; 
    m_normal = normal ;
    setAbsolute(false);
}
void Clipper::setPlane(glm::vec4& absplane)
{
    m_absplane = absplane ;
    setAbsolute(true);
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
    if(m_absolute)
    {
        return m_absplane ; 
    }
    else
    {
        update(model_to_world);
        return m_wplane ; 
    }
}

void Clipper::dump(const char* msg)
{ 
    printf("%s m_mode %d m_absolute %d \n", msg, m_mode, m_absolute);

    print( m_normal,  "m_normal : (model frame) vector normal to plane  ");
    print( m_point,   "m_point  : (model frame) point in the plane ");
    print( m_wnormal, "m_wnormal : (world frame) normalized normal to plane ");
    print( m_wpoint,  "m_wpoint  : (world frame) point in the plane  ");
    print( m_wplane,  "m_wplane  : (world frame) plane equation for shader consumption");
    print( m_absplane,  "m_absplane  : (world frame) absolute input plane equation : for shader consumption");

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

