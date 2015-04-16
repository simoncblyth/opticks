#include "Prog.hh"
#include "Shdr.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include <boost/filesystem.hpp>
#include <iostream>

namespace fs = boost::filesystem;


Prog::Prog(const char* basedir, const char* tag)
{
    setup();

    fs::path tagdir(basedir);
    tagdir /= tag ;

    if(fs::exists(tagdir) && fs::is_directory(tagdir)) 
    {
        examine(tagdir.string().c_str()); 
    }
    else
    {
        LOG(fatal) << "Prog::Prog expected directory at " << tagdir.string() ; 
    }
}


void Prog::setup()
{
    m_names.push_back("vert.glsl");  
    m_names.push_back("geom.glsl");  
    m_names.push_back("frag.glsl");  

    m_codes.push_back(GL_VERTEX_SHADER);
    m_codes.push_back(GL_GEOMETRY_SHADER);
    m_codes.push_back(GL_FRAGMENT_SHADER);
}


void Prog::examine(const char* tagdir)
{
    LOG(info) << "Prog::examine  tag directory at [" << tagdir << "]"  ; 

    for(unsigned int i=0 ; i < m_names.size() ; ++i)
    {
        fs::path glsl(tagdir);
        glsl /= m_names[i] ;

        if(fs::exists(glsl) && fs::is_regular_file(glsl))
        {
            addShader( glsl.string().c_str(), m_codes[i] );
        }
        else
        {
            LOG(warning) << "Prog::examine didnt find file " << glsl.string() ; 
        } 
    }     
}

void Prog::addShader(const char* path, GLenum type )
{
    LOG(info) << "Prog::addShader [" << path << "]" ; 
    Shdr* shdr = new Shdr(path, type ); 
    shdr->Print("Prog::addShader");
}


