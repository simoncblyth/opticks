/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <cassert>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <iomanip>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>

#include "NGLMExt.hpp"
#include "NGLM.hpp"

#ifdef _MSC_VER
// members needs to have dll-interface to be used by clients
#pragma warning( disable : 4251 )
#endif


#include "GLMFormat.hpp"
#include "PLOG.hh"


template <>
const char* GLMType<float>::np_dtype = "np.float32" ; 

template <>
const char* GLMType<double>::np_dtype = "np.float64" ; 

template struct GLMType<float> ;
template struct GLMType<double>;


GLMFormat::GLMFormat(const char* delim, unsigned int prec)
{
    m_delim = delim ; 
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
std::string GLMFormat::format(unsigned int u)
{
    m_ss.str(""); // clear buffer
    m_ss << u ; 
    return m_ss.str();
}




std::string GLMFormat::format(const glm::vec2& v)
{
    std::vector<std::string> vals ; 
    vals.push_back(format(v.x));
    vals.push_back(format(v.y));
    return boost::algorithm::join(vals, ",");
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

std::string GLMFormat::format(const glm::mat4& m)
{
    std::vector<std::string> vals ; 
    vals.push_back(format(m[0]));
    vals.push_back(format(m[1]));
    vals.push_back(format(m[2]));
    vals.push_back(format(m[3]));
    return boost::algorithm::join(vals, " ");
}

std::string GLMFormat::format(const glm::mat3& m)
{
    std::vector<std::string> vals ; 
    vals.push_back(format(m[0]));
    vals.push_back(format(m[1]));
    vals.push_back(format(m[2]));
    return boost::algorithm::join(vals, " ");
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
std::string GLMFormat::format(const glm::uvec4& v)
{
    std::vector<std::string> vals ; 
    vals.push_back(format(v.x));
    vals.push_back(format(v.y));
    vals.push_back(format(v.z));
    vals.push_back(format(v.w));
    return boost::algorithm::join(vals, ",");
}





std::string GLMFormat::format(const glm::ivec3& v)
{
    std::vector<std::string> vals ; 
    vals.push_back(format(v.x));
    vals.push_back(format(v.y));
    vals.push_back(format(v.z));
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


float GLMFormat::float_(const std::string& s )
{
    float f = boost::lexical_cast<float>(s);
    return f ;
}

int GLMFormat::int_(const std::string& s )
{
    int i = boost::lexical_cast<int>(s);
    return i ;
}

unsigned int GLMFormat::uint_(const std::string& s )
{
    unsigned int u = boost::lexical_cast<unsigned int>(s);
    return u ;
}




glm::quat GLMFormat::quat(const std::string& s )
{
    std::vector<std::string> tp; 
    boost::split(tp, s, boost::is_any_of(","));
    assert(tp.size() == 4);

    float w = boost::lexical_cast<float>(tp[0]); 
    float x = boost::lexical_cast<float>(tp[1]); 
    float y = boost::lexical_cast<float>(tp[2]); 
    float z = boost::lexical_cast<float>(tp[3]); 

    glm::quat q(w,x,y,z);
    return q ; 
}


glm::vec2 GLMFormat::vec2(const std::string& s )
{
    std::vector<std::string> tp; 
    boost::split(tp, s, boost::is_any_of(","));
    unsigned int size = tp.size();

    glm::vec2 v(0.f,0.f);
    if(size > 0) v.x = boost::lexical_cast<float>(tp[0]); 
    if(size > 1) v.y = boost::lexical_cast<float>(tp[1]); 

    return v ; 
}


glm::vec3 GLMFormat::vec3(const std::string& s )
{
    std::vector<std::string> tp; 
    boost::split(tp, s, boost::is_any_of(","));
    unsigned int size = tp.size();

    glm::vec3 v(0.f,0.f,0.f);
    if(size > 0) v.x = boost::lexical_cast<float>(tp[0]); 
    if(size > 1) v.y = boost::lexical_cast<float>(tp[1]); 
    if(size > 2) v.z = boost::lexical_cast<float>(tp[2]); 

    return v ; 
}

glm::vec4 GLMFormat::vec4(const std::string& s )
{
    std::vector<std::string> tp; 
    boost::split(tp, s, boost::is_any_of(","));
    unsigned int size = tp.size();

    glm::vec4 v(0.f,0.f,0.f,0.f);
    if(size > 0) v.x = boost::lexical_cast<float>(tp[0]); 
    if(size > 1) v.y = boost::lexical_cast<float>(tp[1]); 
    if(size > 2) v.z = boost::lexical_cast<float>(tp[2]); 
    if(size > 3) v.w = boost::lexical_cast<float>(tp[3]); 

    return v ; 
}




glm::mat4 GLMFormat::mat4(const std::string& s, bool flip )
{
    std::vector<std::string> tp; 

    std::string c(s);

    if(!c.empty())
    {
        boost::trim(c); 
        boost::split(tp, c, boost::is_any_of(m_delim), boost::token_compress_on);
    }

    unsigned int size = tp.size();
    glm::mat4 m;

    if(size > 0)
    {
        for(unsigned int j=0 ; j < 4 ; j++) 
        for(unsigned int k=0 ; k < 4 ; k++) 
        {
            unsigned int offset = j*4+k ;   
            if(offset >= size) break ; 
            float v = boost::lexical_cast<float>(tp[offset]) ;
            if(flip)
            {
                m[j][k] = v ; 
            }
            else
            {
                m[k][j] = v ; 
            }   
        }
    }

    return m ; 
}




glm::mat3 GLMFormat::mat3(const std::string& s, bool flip )
{
    std::vector<std::string> tp; 
    std::string c(s);
    boost::trim(c); 
    boost::split(tp, c, boost::is_any_of(m_delim), boost::token_compress_on);

    unsigned int size = tp.size();

    //LOG(info) <<  "GLMFormat::mat3 size " << size ; 
    //for(unsigned int i=0 ; i < size ; i++) LOG(info) << std::setw(4) << i << " [" << tp[i] << "]" << std::endl  ; 

    glm::mat3 m;
    for(unsigned int j=0 ; j < 3 ; j++) 
    for(unsigned int k=0 ; k < 3 ; k++) 
    {
        unsigned int offset = j*3+k ;   
        if(offset >= size) break ; 
        float v = boost::lexical_cast<float>(tp[offset]) ;
        if(flip)
        {
            m[j][k] = v ; 
        }
        else
        {
            m[k][j] = v ; 
        }   
    }
    return m ; 
}









glm::ivec4 GLMFormat::ivec4(const std::string& s )
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

glm::uvec4 GLMFormat::uvec4(const std::string& s )
{
    std::vector<std::string> tp; 
    boost::split(tp, s, boost::is_any_of(","));

    unsigned int size = tp.size();

    glm::uvec4 v(0,0,0,0) ;  
    if(size > 0) v.x = boost::lexical_cast<unsigned int>(tp[0]); 
    if(size > 1) v.y = boost::lexical_cast<unsigned int>(tp[1]); 
    if(size > 2) v.z = boost::lexical_cast<unsigned int>(tp[2]); 
    if(size > 3) v.w = boost::lexical_cast<unsigned int>(tp[3]); 

    return v ; 
}





glm::ivec3 GLMFormat::ivec3(const std::string& s )
{
    std::vector<std::string> tp; 
    boost::split(tp, s, boost::is_any_of(","));

    unsigned int size = tp.size();

    glm::ivec3 v(0,0,0) ;  
    if(size > 0) v.x = boost::lexical_cast<int>(tp[0]); 
    if(size > 1) v.y = boost::lexical_cast<int>(tp[1]); 
    if(size > 2) v.z = boost::lexical_cast<int>(tp[2]); 

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
std::string gformat(unsigned int i)
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
std::string gformat(const glm::uvec4& v )
{
    GLMFormat fmt; 
    return fmt.format(v);
}


std::string gformat(const glm::ivec3& v )
{
    GLMFormat fmt; 
    return fmt.format(v);
}




std::string gformat(const glm::quat& q )
{
    GLMFormat fmt; 
    return fmt.format(q);
}

std::string gformat(const glm::mat4& m )
{
    GLMFormat fmt; 
    return nglmext::is_identity(m) ? "Id" :  fmt.format(m) ;
}

std::string gformat(const glm::mat3& m )
{
    GLMFormat fmt; 
    return fmt.format(m);
}


std::string gpresent_label(const char* label, unsigned lwid)
{
    std::stringstream ss ; 
    ss << std::setw(lwid) << ( label ? label : " " ) ; 
    return ss.str();
}





std::string gfromstring(const glm::mat4& m, bool flip)
{
    std::stringstream ss ; 

    ss << "np.fromstring("  ;
    ss << "\"" ;

    for(int i=0 ; i < 4 ; i++)
    for(int j=0 ; j < 4 ; j++)
        ss << ( flip ? m[j][i] : m[i][j] ) << " " ; 
        
    ss << "\"" ;
    ss << ", dtype=np.float32, sep=\" \").reshape(4,4) " ;

    return ss.str();
}



template<typename T>
std::string gfromstring_(const glm::tmat4x4<T>& m, bool flip)
{
    std::stringstream ss ; 

    ss << "np.fromstring("  ;
    ss << "\"" ;

    for(int i=0 ; i < 4 ; i++)
    for(int j=0 ; j < 4 ; j++)
        ss << ( flip ? m[j][i] : m[i][j] ) << " " ; 
        
    ss << "\"" ;
    ss << ", dtype=" << GLMType<T>::np_dtype  << ", sep=\" \").reshape(4,4) " ;

    return ss.str();
}








std::string gpresent(const char* label, const glm::mat4& m, unsigned prec, unsigned wid, unsigned lwid, bool flip )
{
    std::stringstream ss ; 
    for(int i=0 ; i < 4 ; i++)
    {
        ss << std::setw(lwid) << ( i == 0 && label ? label  : " " ) ; 
        for(int j=0 ; j < 4 ; j++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) << ( flip ? m[j][i] : m[i][j] ) << " " ; 
        ss << std::endl ; 
    }
    ss << gfromstring(m, flip) <<  std::endl ; 
    return ss.str();
}




NPY_API std::string gpresent__(const char* label, const glm::tmat4x4<float>& m, unsigned prec, unsigned wid, unsigned lwid, bool flip )
{
    std::stringstream ss ; 
    for(int i=0 ; i < 4 ; i++)
    {
        ss << std::setw(lwid) << ( i == 0 && label ? label  : " " ) ; 
        for(int j=0 ; j < 4 ; j++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) << ( flip ? m[j][i] : m[i][j] ) << " " ; 
        ss << std::endl ; 
    }
    ss << gfromstring_<float>(m, flip) <<  std::endl ; 
    return ss.str();
}

NPY_API std::string gpresent__(const char* label, const glm::tmat4x4<double>& m, unsigned prec, unsigned wid, unsigned lwid, bool flip )
{
    std::stringstream ss ; 
    for(int i=0 ; i < 4 ; i++)
    {
        ss << std::setw(lwid) << ( i == 0 && label ? label  : " " ) ; 
        for(int j=0 ; j < 4 ; j++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) << ( flip ? m[j][i] : m[i][j] ) << " " ; 
        ss << std::endl ; 
    }
    ss << gfromstring_<double>(m, flip) <<  std::endl ; 
    return ss.str();
}






std::string gpresent(const char* label, const glm::mat3& m, unsigned prec, unsigned wid, unsigned lwid, bool flip )
{
    std::stringstream ss ; 
    for(int i=0 ; i < 3 ; i++)
    {
        ss << std::setw(lwid) << ( i == 0 ? label : " " ) ; 
        for(int j=0 ; j < 3 ; j++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) << ( flip ? m[j][i] : m[i][j] )  << " " ; 
        ss << std::endl ; 
    }
    return ss.str();
}





std::string gpresent(const char* label, const glm::ivec3& m, unsigned prec, unsigned wid, unsigned lwid  )
{
    std::stringstream ss ; 
    ss << std::setw(lwid) << label ; 
    for(int i=0 ; i < 3 ; i++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) << m[i]  << " " ; 
    ss << std::endl ; 
    return ss.str();
}


std::string gpresent(const char* label, const glm::vec3& m, unsigned prec, unsigned wid, unsigned lwid  )
{
    std::stringstream ss ; 
    ss << std::setw(lwid) << label ; 
    for(int i=0 ; i < 3 ; i++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) << m[i]  << " " ; 
    ss << std::endl ; 
    return ss.str();
}

std::string gpresent(const char* label, const glm::vec4& m, unsigned prec, unsigned wid, unsigned lwid )
{
    std::stringstream ss ; 
    ss << std::setw(lwid) << label ; 
    for(int i=0 ; i < 4 ; i++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) << m[i]  << " " ; 
    ss << std::endl ; 
    return ss.str();
}

std::string gpresent(const char* label, const glm::ivec4& m, unsigned prec, unsigned wid, unsigned lwid  )
{
    std::stringstream ss ; 
    ss << std::setw(lwid) << label ; 
    for(int i=0 ; i < 4 ; i++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) << m[i]  << " " ; 
    ss << std::endl ; 
    return ss.str();
}

std::string gpresent(const char* label, const glm::uvec4& m, unsigned prec, unsigned wid, unsigned lwid  )
{
    std::stringstream ss ; 
    ss << std::setw(lwid) << label ; 
    for(int i=0 ; i < 4 ; i++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) << m[i]  << " " ; 
    ss << std::endl ; 
    return ss.str();
}




std::string gpresent_(const char* label, const glm::vec4& m, unsigned prec, unsigned wid, unsigned lwid )
{
    std::stringstream ss ; 
    ss << std::setw(lwid) << label ; 
    for(int i=0 ; i < 4 ; i++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) << m[i]  << " " ; 
    return ss.str();
}






std::string gpresent(const glm::vec3& v, unsigned int prec, unsigned int wid)
{
    std::stringstream ss ; 

    ss << "( "
       << std::setprecision(prec) << std::fixed 
       << std::setw(wid) << v.x 
       << std::setw(wid) << v.y 
       << std::setw(wid) << v.z
       << ")" ;

    return ss.str();
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

std::string gpresent(const glm::vec2& v, unsigned int prec, unsigned int wid)
{
    std::stringstream ss ; 

    ss 
       << std::setprecision(prec) << std::fixed 
       << std::setw(wid) << v.x 
       << std::setw(wid) << v.y 
       ;

    return ss.str();
}




std::string gpresent(const glm::ivec4& v, unsigned wid)
{
    std::stringstream ss ; 
    ss 
       << std::setw(wid) << v.x 
       << " "
       << std::setw(wid) << v.y 
       << " "
       << std::setw(wid) << v.z
       << " "
       << std::setw(wid) << v.w 
       ;

    return ss.str();

}

std::string gpresent(const glm::ivec4& v, unsigned wid_x, unsigned wid_y, unsigned wid_z, unsigned wid_w)
{
    std::stringstream ss ; 
    ss 
       << std::setw(wid_x) << v.x 
       << " "
       << std::setw(wid_y) << v.y 
       << " "
       << std::setw(wid_z) << v.z
       << " "
       << std::setw(wid_w) << v.w 
       ;

    return ss.str();
}



std::string gpresent(const glm::uvec4& v, unsigned wid)
{
    std::stringstream ss ; 
    ss 
       << std::setw(wid) << v.x 
       << " "
       << std::setw(wid) << v.y 
       << " "
       << std::setw(wid) << v.z
       << " "
       << std::setw(wid) << v.w 
       ;
    return ss.str();
}


std::string gpresent(const glm::uvec4& v, unsigned wid_x, unsigned wid_y, unsigned wid_z, unsigned wid_w)
{
    std::stringstream ss ; 
    ss 
       << std::setw(wid_x) << v.x 
       << " "
       << std::setw(wid_y) << v.y 
       << " "
       << std::setw(wid_z) << v.z
       << " "
       << std::setw(wid_w) << v.w 
       ;
    return ss.str();
}





std::string gpresent(const glm::uvec3& v, unsigned wid)
{
    std::stringstream ss ; 
    ss 
       << std::setw(wid) << v.x 
       << " "
       << std::setw(wid) << v.y 
       << " "
       << std::setw(wid) << v.z
       ;

    return ss.str();
}






float gfloat_(const std::string&s )
{
    GLMFormat fmt; 
    return fmt.float_(s);
}
int gint_(const std::string&s )
{
    GLMFormat fmt; 
    return fmt.int_(s);
}
unsigned int guint_(const std::string&s )
{
    GLMFormat fmt; 
    return fmt.uint_(s);
}




glm::vec2 gvec2(const std::string& s )
{
    GLMFormat fmt; 
    return fmt.vec2(s);
}
glm::vec3 gvec3(const std::string& s )
{
    GLMFormat fmt; 
    return fmt.vec3(s);
}

glm::vec4 gvec4(const std::string& s )
{
    GLMFormat fmt; 
    return fmt.vec4(s);
}
glm::ivec4 givec4(const std::string& s )
{
    GLMFormat fmt; 
    return fmt.ivec4(s);
}
glm::uvec4 guvec4(const std::string& s )
{
    GLMFormat fmt; 
    return fmt.uvec4(s);
}

glm::ivec3 givec3(const std::string& s )
{
    GLMFormat fmt; 
    return fmt.ivec3(s);
}
glm::quat gquat(const std::string& s )
{
    GLMFormat fmt; 
    return fmt.quat(s);
}
glm::mat4 gmat4(const std::string& s, bool flip, const char* delim)
{
    GLMFormat fmt(delim); 
    return fmt.mat4(s, flip);
}
glm::mat3 gmat3(const std::string& s, bool flip, const char* delim)
{
    GLMFormat fmt(delim); 
    return fmt.mat3(s, flip);
}



NPY_API std::string gfromstring_(const glm::tmat4x4<float>& m, bool flip) ;
NPY_API std::string gfromstring_(const glm::tmat4x4<double>& m, bool flip) ;

