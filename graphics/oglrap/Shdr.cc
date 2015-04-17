#include "Shdr.hh"

#include <GL/glew.h>


#include "string.h"

#include <iostream>
#include <iomanip>
#include <fstream>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



Shdr::Shdr(const char* path, GLenum type, bool live)
    :
    m_path(strdup(path)),
    m_type(type),
    m_id(0),
    m_live(live)
{
    readFile(m_path);
}


void Shdr::createAndCompile()
{
    if(!m_live) return ;
 
    m_id = glCreateShader(m_type);

    const char* content_c = m_content.c_str();

    glShaderSource (m_id, 1, &content_c, NULL);

    glCompileShader (m_id);

    int params = -1;

    glGetShaderiv (m_id, GL_COMPILE_STATUS, &params);

    if (GL_TRUE != params) 
    {
        fprintf (stderr, "ERROR: GL shader index %i did not compile\n", m_id );

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




void Shdr::readFile(const char* path)
{
    LOG(debug) << "Shdr::readFile " << path << std::endl ; 

    std::ifstream fs(path, std::ios::in);
    if(!fs.is_open()) 
    {
        LOG(fatal) << "Shdr::readFile failed to open " << path ; 
        return ;
    }   

    std::string line = ""; 
    while(!fs.eof()) 
    {
        std::getline(fs, line);
        m_content.append(line + "\n");
        m_lines.push_back(line);
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



