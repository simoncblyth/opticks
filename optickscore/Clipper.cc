
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "BLog.hh"

// npy-
#include "NGLM.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

// okc-
#include "Clipper.hh"


const char* Clipper::PREFIX = "clipper" ;
const char* Clipper::getPrefix()
{
   return PREFIX ; 
}



const char* Clipper::CUTMODE    = "cutmode" ;
const char* Clipper::CUTPOINT   = "cutpoint" ;
const char* Clipper::CUTNORMAL  = "cutnormal" ;
const char* Clipper::CUTPLANE   = "cutplane" ;
const char* Clipper::CUTPRINT   = "cutprint" ;



// Configurable
bool Clipper::accepts(const char* name)
{
    return 
         strcmp(name,CUTMODE)==0   ||
         strcmp(name,CUTPOINT)==0  ||
         strcmp(name,CUTNORMAL)==0 ||
         strcmp(name,CUTPLANE)==0  ||
         strcmp(name,CUTPRINT)==0  ;
}

std::vector<std::string> Clipper::getTags()
{
    std::vector<std::string> tags ;
//    tags.push_back(CUTMODE);
    tags.push_back(CUTPOINT);
    tags.push_back(CUTNORMAL);
    tags.push_back(CUTPLANE);
//    tags.push_back(CUTPRINT);
    return tags ; 
}


Clipper::Clipper() :
   m_mode(-1),
   m_absolute(false), 
   m_point(0,0,0),
   m_normal(1,0,0),
   m_absplane(1,0,0,1),   // placeholder 
   m_float3(NULL)
{
   m_float3 = new float[3];
   m_float3[0] = 0.1f ;  
   m_float3[1] = 0.2f ;  
   m_float3[2] = 0.3f ;  
}


glm::vec3& Clipper::getPoint()
{
    return m_point ; 
}
glm::vec3& Clipper::getNormal()
{
    return m_normal ; 
}
glm::vec4& Clipper::getPlane()
{
    return m_absplane ; 
}


int Clipper::getMode()
{
    return m_mode ; 
}

void Clipper::next()
{
    // Interactor invokes this on pressing C, for now just toggle between -1 and 0
    m_mode = m_mode != -1 ? -1 : 0 ; 
}






void Clipper::configure(const char* name, const char* value_)
{
    std::string value(value_);
    set(name, value);
}



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


std::string Clipper::get(const char* name)
{
    std::string s ; 
    if(strcmp(name,CUTPOINT)==0)
    {
        glm::vec3 v = getPoint();
        s = gformat(v);
    } 
    else if(strcmp(name,CUTNORMAL)==0)
    {
        glm::vec3 v = getNormal();
        s = gformat(v);
    }
    else if(strcmp(name,CUTPLANE)==0)
    {
        glm::vec4 v = getPlane();
        s = gformat(v);
    }
    else
         printf("Clipper::get bad name %s\n", name);

    return s;
}



void Clipper::set(const char* name, std::string& arg_)
{
    std::vector<std::string> arg;
    boost::split(arg, arg_, boost::is_any_of(","));

    if(arg.size() == 3 )
    {
        glm::vec3 v = gvec3(arg_);

        if(     strcmp(name,CUTPOINT)==0)     setPoint(v);
        else if(strcmp(name,CUTNORMAL) == 0 ) setNormal(v);
        else
              printf("Clipper::configureS bad name %s\n", name);
    }
    else if(arg.size() == 4)
    {
        glm::vec4 v = gvec4(arg_);

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
    LOG(debug)<<"Clipper::setAbsolute m_absolute -> absolute " << m_absolute << " -> " << absolute ; 
    m_absolute = absolute ;
}

void Clipper::setPoint(glm::vec3& point)
{
    LOG(debug)<<"Clipper::setPoint m_abolute: " << m_absolute ; 
    m_point = point ;
    setAbsolute(false);
}
void Clipper::setNormal(glm::vec3& normal)
{
    LOG(debug)<<"Clipper::setNormal m_abolute: " << m_absolute ; 
    m_normal = normal ;
    setAbsolute(false);
}
void Clipper::setPlane(glm::vec4& absplane)
{
    m_absplane = absplane ;
    setAbsolute(true);
}


float* Clipper::getPointPtr()
{
    return glm::value_ptr(m_point);
}
float* Clipper::getNormalPtr()
{
    return glm::value_ptr(m_normal);
}
float* Clipper::getPlanePtr()
{
    return glm::value_ptr(m_absplane);
}


void Clipper::update(glm::mat4& model_to_world)
{
    // transform point position and normal direction from model to world frame
    //
    // model_to_world does uniform extent scaling and a translation only
    // so does not change directions

    m_normal = glm::normalize( m_normal );

    m_wnormal = m_normal ;

    m_wpoint = glm::vec3( model_to_world * glm::vec4(m_point,  1.f));

    m_wplane = glm::vec4( m_wnormal, -glm::dot( m_wpoint, m_wnormal ));  // plane equation for OpenGL 
}


glm::vec4& Clipper::getClipPlane(glm::mat4& model_to_world)
{
    //dump("Clipper::getClipPlane");
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

    print( getPointPtr(),  "getPointPtr()", 3 );
    print( getNormalPtr(), "getNormalPtr()", 3 );
    print( getPlanePtr(),  "getPlanePtr()", 4 );
    print( m_float3  ,  "m_float3", 3 );

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

