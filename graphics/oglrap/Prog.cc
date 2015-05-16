#include "Prog.hh"
#include "Shdr.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include <boost/filesystem.hpp>
#include <iostream>

namespace fs = boost::filesystem;

const char* GL_type_to_string (GLenum type);


Prog::Prog(const char* basedir, const char* tag, bool live) :
   m_live(live),
   m_tagdir(NULL)
{
    setup();

    fs::path tagdir(basedir);
    tagdir /= tag ;

    m_tagdir = strdup(tagdir.string().c_str()); 

    if(fs::exists(tagdir) && fs::is_directory(tagdir)) 
    {
        readSources(m_tagdir);
    }
    else
    {
        LOG(fatal) << "Prog::Prog expected directory at " << m_tagdir ; 
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


void Prog::readSources(const char* tagdir)
{
    LOG(debug) << "Prog::examine  tag directory at [" << tagdir << "]"  ; 

    for(unsigned int i=0 ; i < m_names.size() ; ++i)
    {
        fs::path glsl(tagdir);
        glsl /= m_names[i] ;

        if(fs::exists(glsl) && fs::is_regular_file(glsl))
        {
            const char* path = glsl.string().c_str();
            GLenum type = m_codes[i] ;
            Shdr* shdr = new Shdr(path, type, m_live); 
            m_shaders.push_back(shdr);
        }
        else
        {
            LOG(debug) << "Prog::examine didnt find file " << glsl.string() ; 
        } 
    }     
}

void Prog::createAndLink()
{
    LOG(debug) << "Prog::createAndLink tagdir " << m_tagdir ; 

    m_id = glCreateProgram();

    for(unsigned int i=0 ; i<m_shaders.size() ; i++)
    {
        Shdr* shdr = m_shaders[i];
        shdr->createAndCompile();           
        glAttachShader (m_id, shdr->getId());
    }
    link();
    validate();
    collectLocations();
    //Print("Prog::createAndLink");
}

void Prog::link()
{
    glLinkProgram (m_id);

    int params = -1;
    glGetProgramiv (m_id, GL_LINK_STATUS, &params);

    if (GL_TRUE != params) 
    {
        fprintf (stderr, "ERROR: linking GL shader program index %i \n", m_id);
        _print_program_info_log();
        Print("Prog::link ERROR"); 
        exit(1); 
    } 
}

void Prog::validate()
{
    glValidateProgram (m_id);

    int params = -1;
    glGetProgramiv (m_id, GL_VALIDATE_STATUS, &params);

    if (GL_TRUE != params) 
    {
        _print_program_info_log();
        Print("Prog::validate ERROR"); 
        LOG(fatal) << "Prog::validate failure for tagdir [" << m_tagdir << "]"  ; 
        exit(1);
    }
}

void Prog::collectLocations()
{
    bool print = false ; 
    traverseActive(Attribute, print);
    traverseActive(Uniform, print);
}


GLint Prog::attribute(const char* name_, bool required)
{
    std::string name(name_);
    if(m_attributes.find(name) == m_attributes.end())
    {
         if(required)
         {
             LOG(fatal) << "Prog::attribute " << m_tagdir << " failed to find required attribute [" << name << "]" ;
             exit(1);
         }
         else
         {
             LOG(debug) << "Prog::attribute " << m_tagdir << " did not find optional attribute [" << name << "]" ;
         }
         return -1 ; 
    }
    return m_attributes[name];
}

GLint Prog::uniform(const char* name_, bool required)
{
    std::string name(name_);
    if(m_uniforms.find(name) == m_uniforms.end())
    {
         if(required)
         {
             LOG(fatal) << "Prog::uniform " << m_tagdir << " failed to find required uniform [" << name << "]" ;
             exit(1);
         }
         else
         {
             LOG(debug) << "Prog::uniform " << m_tagdir << " did not find optional uniform [" << name << "]" ;
         }
         return -1 ; 
    }
    return m_uniforms[name];
}



void Prog::_print_program_info_log()
{
    int max_length = 2048;
    int actual_length = 0;
    char log[2048];

    glGetProgramInfoLog (m_id, max_length, &actual_length, log);

    printf ("Prog::_print_program_info_log id %u:\n%s", m_id, log);
}


void Prog::Summary(const char* msg)
{
    LOG(info) << msg << "\n" << m_tagdir ;  
    dumpUniforms();
    dumpAttributes();
}


void Prog::Print(const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i<m_shaders.size() ; i++)
    {
        Shdr* shdr = m_shaders[i];
        shdr->Print(msg);           
    }
    dumpUniforms();
    dumpAttributes();
    printStatus();
}


void Prog::printStatus()
{
  int params = -1;

  glGetProgramiv (m_id, GL_LINK_STATUS, &params);
  printf ("GL_LINK_STATUS = %i\n", params);
  
  glGetProgramiv (m_id, GL_ATTACHED_SHADERS, &params);
  printf ("GL_ATTACHED_SHADERS = %i\n", params);
  
  glGetProgramiv (m_id, GL_ACTIVE_ATTRIBUTES, &params);
  printf ("GL_ACTIVE_ATTRIBUTES = %i\n", params);

  glGetProgramiv (m_id, GL_ACTIVE_UNIFORMS, &params);
  printf ("GL_ACTIVE_UNIFORMS = %i\n", params);
 
  glGetProgramiv (m_id, GL_VALIDATE_STATUS, &params);
  printf ("GL_VALIDATE_STATUS = %i\n", params);

}





void Prog::traverseActive(Obj_t obj, bool print)
{
    int params = -1;
    switch(obj)
    {
       case Attribute: 
                      glGetProgramiv (m_id, GL_ACTIVE_ATTRIBUTES, &params); 
                      break ;
       case   Uniform: 
                      glGetProgramiv (m_id, GL_ACTIVE_UNIFORMS, &params); 
                      break ;
    }  

    for(int i = 0; i < params; i++) 
    {
        char name[64];
        int max_length = 64;
        int actual_length = 0;
        int size = 0;
        GLenum type;

        switch(obj)
        {
            case Attribute: 
                     glGetActiveAttrib(m_id, i, max_length, &actual_length, &size, &type, name );
                     break;
            case Uniform: 
                     glGetActiveUniform(m_id, i, max_length, &actual_length, &size, &type, name );
                     break;
        }

        if(size > 1) 
        {
            for(int j = 0; j < size; j++)
            {
                char long_name[64];
                sprintf (long_name, "%s[%i]", name, j);
                traverseLocation(obj, type, long_name, print);
            }
        } 
        else 
        {
            traverseLocation(obj, type, name, print);
        }
    }
}


void Prog::dumpUniforms()
{
    for(ObjMap_t::iterator it=m_uniforms.begin() ; it!=m_uniforms.end() ; it++)
    {
        std::string name = it->first ;
        GLuint location = it->second ;
        GLenum type = m_utype[name];
        printf(".  %c%d  %20s %20s \n", 'U', location, GL_type_to_string(type),name.c_str());
    }
}   
    
void Prog::dumpAttributes()
{
    for(ObjMap_t::iterator it=m_attributes.begin() ; it!=m_attributes.end() ; it++)
    {
        std::string name = it->first ;
        GLuint location = it->second ;
        GLenum type = m_atype[name];
        printf(".  %c%d  %20s %20s \n", 'A', location, GL_type_to_string(type),name.c_str());
    }
}   
 

void Prog::traverseLocation(Obj_t obj, GLenum type,  const char* name_, bool print)
{
    std::string name(name_);

    char t('?') ; 
    GLint location(-1) ;
    switch(obj)
    {
        case Attribute: 
             t = 'A' ;
             location = glGetAttribLocation (m_id, name_);
             assert(location > -1);
             m_attributes[name] = location ;
             m_atype[name] = type ;
             break ;
        case Uniform: 
             t = 'U' ;
             location = glGetUniformLocation (m_id, name_);
             assert(location > -1);
             m_uniforms[name] = location ;
             m_utype[name] = type ;
             break ;
    }

    if(print)
    {
        printf("  %c%d  %20s %20s \n", t, location, GL_type_to_string(type),name_);
    }
}




const char* Prog::GL_type_to_string(GLenum type) 
{
    switch (type) 
    {
        case GL_BOOL: return "bool";
        case GL_INT: return "int";
        case GL_FLOAT: return "float";
        case GL_FLOAT_VEC2: return "vec2";
        case GL_FLOAT_VEC3: return "vec3";
        case GL_FLOAT_VEC4: return "vec4";
        case GL_FLOAT_MAT2: return "mat2";
        case GL_FLOAT_MAT3: return "mat3";
        case GL_FLOAT_MAT4: return "mat4";
        case GL_SAMPLER_2D: return "sampler2D";
        case GL_SAMPLER_3D: return "sampler3D";
        case GL_SAMPLER_CUBE: return "samplerCube";
        case GL_SAMPLER_2D_SHADOW: return "sampler2DShadow";
        default: break;
    }
    return "other";
}


