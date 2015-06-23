#include "GLMFormat.hpp"

#include "assert.h"
#include "stdio.h"
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>  

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>

#include <vector>
#include <iomanip>


GLMFormat::GLMFormat(unsigned int prec)
{
    m_ss.precision(prec) ; 
    m_ss << std::fixed ; 
}

std::string GLMFormat::format(float f)
{
    m_ss.str(""); // clear buffer
    m_ss << f ; 
    return m_ss.str();
}
std::string GLMFormat::format(int i)
{
    m_ss.str(""); // clear buffer
    m_ss << i ; 
    return m_ss.str();
}
std::string GLMFormat::format(const glm::vec3& v)
{
    std::vector<std::string> vals ; 
    vals.push_back(format(v.x));
    vals.push_back(format(v.y));
    vals.push_back(format(v.z));
    return boost::algorithm::join(vals, ",");
}
std::string GLMFormat::format(const glm::vec4& v)
{
    std::vector<std::string> vals ; 
    vals.push_back(format(v.x));
    vals.push_back(format(v.y));
    vals.push_back(format(v.z));
    vals.push_back(format(v.w));
    return boost::algorithm::join(vals, ",");
}




std::string GLMFormat::format(const glm::ivec4& v)
{
    std::vector<std::string> vals ; 
    vals.push_back(format(v.x));
    vals.push_back(format(v.y));
    vals.push_back(format(v.z));
    vals.push_back(format(v.w));
    return boost::algorithm::join(vals, ",");
}
std::string GLMFormat::format(const glm::quat& q)
{
    std::vector<std::string> vals ; 
    vals.push_back(format(q.w));
    vals.push_back(format(q.x));
    vals.push_back(format(q.y));
    vals.push_back(format(q.z));
    return boost::algorithm::join(vals, ",");
}


float GLMFormat::float_(std::string& s )
{
    float f = boost::lexical_cast<float>(s);
    return f ;
}

int GLMFormat::int_(std::string& s )
{
    int i = boost::lexical_cast<int>(s);
    return i ;
}


glm::quat GLMFormat::quat(std::string& s )
{
    std::vector<std::string> tp; 
    boost::split(tp, s, boost::is_any_of(","));

    float w = boost::lexical_cast<float>(tp[0]); 
    float x = boost::lexical_cast<float>(tp[1]); 
    float y = boost::lexical_cast<float>(tp[2]); 
    float z = boost::lexical_cast<float>(tp[3]); 

    glm::quat q(w,x,y,z);
    return q ; 
}

glm::vec3 GLMFormat::vec3(std::string& s )
{
    std::vector<std::string> tp; 
    boost::split(tp, s, boost::is_any_of(","));
    assert(tp.size() == 3);

    float x = boost::lexical_cast<float>(tp[0]); 
    float y = boost::lexical_cast<float>(tp[1]); 
    float z = boost::lexical_cast<float>(tp[2]); 

    glm::vec3 v(x,y,z);
    return v ; 
}

glm::vec4 GLMFormat::vec4(std::string& s )
{
    std::vector<std::string> tp; 
    boost::split(tp, s, boost::is_any_of(","));
    assert(tp.size() == 4);

    float x = boost::lexical_cast<float>(tp[0]); 
    float y = boost::lexical_cast<float>(tp[1]); 
    float z = boost::lexical_cast<float>(tp[2]); 
    float w = boost::lexical_cast<float>(tp[3]); 

    glm::vec4 v(x,y,z,w);
    return v ; 
}

glm::ivec4 GLMFormat::ivec4(std::string& s )
{
    std::vector<std::string> tp; 
    boost::split(tp, s, boost::is_any_of(","));

    unsigned int size = tp.size();

    glm::ivec4 v(0,0,0,0) ;  
    if(size > 0) v.x = boost::lexical_cast<int>(tp[0]); 
    if(size > 1) v.y = boost::lexical_cast<int>(tp[1]); 
    if(size > 2) v.z = boost::lexical_cast<int>(tp[2]); 
    if(size > 3) v.w = boost::lexical_cast<int>(tp[3]); 

    return v ; 
}




std::string gformat(float f)
{
    GLMFormat fmt; 
    return fmt.format(f);
}
std::string gformat(int i)
{
    GLMFormat fmt; 
    return fmt.format(i);
}
std::string gformat(const glm::vec3& v )
{
    GLMFormat fmt; 
    return fmt.format(v);
}
std::string gformat(const glm::vec4& v )
{
    GLMFormat fmt; 
    return fmt.format(v);
}
std::string gformat(const glm::ivec4& v )
{
    GLMFormat fmt; 
    return fmt.format(v);
}
std::string gformat(const glm::quat& q )
{
    GLMFormat fmt; 
    return fmt.format(q);
}


std::string gpresent(const glm::vec4& v, unsigned int prec, unsigned int wid)
{
    std::stringstream ss ; 

    ss << "( "
       << std::setprecision(prec) << std::fixed 
       << std::setw(wid) << v.x 
       << std::setw(wid) << v.y 
       << std::setw(wid) << v.z
       << "  " 
       << std::setw(wid) << v.w 
       << ")" ;

    return ss.str();
}







float gfloat_(std::string&s )
{
    GLMFormat fmt; 
    return fmt.float_(s);
}
int gint_(std::string&s )
{
    GLMFormat fmt; 
    return fmt.int_(s);
}
glm::vec3 gvec3(std::string& s )
{
    GLMFormat fmt; 
    return fmt.vec3(s);
}
glm::vec4 gvec4(std::string& s )
{
    GLMFormat fmt; 
    return fmt.vec4(s);
}
glm::ivec4 givec4(std::string& s )
{
    GLMFormat fmt; 
    return fmt.ivec4(s);
}
glm::quat gquat(std::string& s )
{
    GLMFormat fmt; 
    return fmt.quat(s);
}





