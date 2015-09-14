#include "Shdr.hh"

#include <GL/glew.h>


#include "string.h"

#include <iostream>
#include <iomanip>
#include <fstream>


#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

const char* Shdr::incl_prefix = "#incl" ;  // not "#include" to remind that this is non-standard

Shdr::Shdr(const char* path, GLenum type, const char* incl_path )
    :
    m_path(strdup(path)),
    m_type(type),
    m_id(0)
{
    setInclPath(incl_path);
    readFile(m_path);
}


void Shdr::createAndCompile()
{
    m_id = glCreateShader(m_type);

    const char* content_c = m_content.c_str();

    glShaderSource (m_id, 1, &content_c, NULL);

    glCompileShader (m_id);

    int params = -1;

    glGetShaderiv (m_id, GL_COMPILE_STATUS, &params);

    if (GL_TRUE != params) 
    {
        LOG(fatal) << "Shdr::createAndCompile FAILED " << m_path  ; 

        _print_shader_info_log();

        exit(1); 
    } 
}



void Shdr::_print_shader_info_log() 
{
    int max_length = 2048;
    int actual_length = 0;
    char log[2048];

    glGetShaderInfoLog(m_id, max_length, &actual_length, log);

    printf ("shader info log for GL index %u:\n%s\n", m_id, log);
}



void Shdr::setInclPath(const char* incl_path)
{
    boost::split(m_incl_dirs,incl_path,boost::is_any_of(":"));
}


std::string Shdr::resolve(const char* name)
{ 
    std::string path ; 
    for(unsigned int i=0 ; i < m_incl_dirs.size() ; i++)
    {
        fs::path candidate(m_incl_dirs[i]) ;
        candidate /= name ;
        if(fs::exists(candidate) && fs::is_regular_file(candidate))
        {
            path = candidate.string() ;
            break ;  
        }
    }
    return path ;  
} 

void Shdr::readFile(const char* path)
{
    LOG(debug) << "Shdr::readFile " << path << std::endl ; 

    std::ifstream fs(path, std::ios::in);
    if(!fs.is_open()) 
    {
        LOG(fatal) << "Shdr::readFile failed to open " << path ; 
        return ;
    }   

    
    std::string incl_name = "" ; 
    std::string line = ""; 
    while(!fs.eof()) 
    {
        std::getline(fs, line);
        
        if(strncmp(line.c_str(), incl_prefix, strlen(incl_prefix))==0)
        {
            // TODO: make plucking of the incl path more robust using regexsearch, see bregex-/regex_extract_quoted
            incl_name = line.c_str() + strlen(incl_prefix) + 1 ; 
            std::string incl_path = resolve(incl_name.c_str());

            if(incl_path.empty())
            {
                LOG(fatal) << "Shdr::readFile FATAL for "
                             << m_path 
                             << " failed to resolve #incl [" << incl_name << "]"  ;
                for(unsigned int i=0 ; i < m_incl_dirs.size() ; i++) LOG(warning) << "Shdr::readFile [" << i << "][" << m_incl_dirs[i] << "]" ; 
                assert(0);    
            }
            else
            {
                 LOG(debug) << "Shdr::readFile " << incl_path ; 
                 readFile(incl_path.c_str());
            }
        }
        else
        { 
            m_content.append(line + "\n");
            m_lines.push_back(line);
        }
    }   
    fs.close();
}


void Shdr::Print(const char* msg)
{
    std::cout << msg << " " << m_path << " linecount " << m_lines.size() << std::endl ; 
    for(unsigned int i=0 ; i<m_lines.size() ; ++i )
    {
        std::cout << std::setw(3) << i << " : "    
                  << m_lines[i] << std::endl ; 
    }
}



