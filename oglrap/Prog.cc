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

#include <cstdio>
#include <iostream>
#include <sstream>

#include "Prog.hh"
#include "ProgLog.hh"
#include "Shdr.hh"
#include "G.hh"

#include "BFile.hh"
#include "PLOG.hh"

// https://www.khronos.org/opengl/wiki/Shader_Compilation


const char* GL_type_to_string (GLenum type);


GLuint Prog::getId()
{
    return m_id ; 
}

void Prog::setInclPath(const char* path)
{   
    if(!path) return ;
    m_incl_path = path ; 
}

void Prog::setVerbosity(unsigned verbosity)
{   
    m_verbosity = verbosity ; 
}

void Prog::setNoFrag(bool nofrag)
{   
    m_nofrag = nofrag ; 
}



std::string Prog::desc() const 
{
    std::stringstream ss ; 

    ss << " Prog " 
       << " tag:" << m_tag
       << " verbosity:" << m_verbosity 
       ;

    return ss.str();
}



Prog::Prog(const char* basedir, const char* tag, const char* incl_path, bool ubo) 
    :
    m_tagdir(NULL), 
    m_tag(strdup(tag)),
    m_ubo(ubo),
    m_verbosity(0)
{
    if(basedir == NULL )
    {
        LOG(fatal) << "Prog::Prog [" << tag << "] cannot read sources as no basedir provided " ; 
        return ; 
    }

    setInclPath(incl_path);
    setup();

    std::string tagdir = BFile::FormPath(basedir, tag);
    if(BFile::ExistsNativeDir(tagdir)) 
    {
        m_tagdir = strdup(tagdir.c_str());
        readSources(m_tagdir);
    }
    else
    {
        LOG(fatal) << "Prog::Prog MISSING directory"
                   << " tagd " << ( tag ? tag : "-" ) 
                   << " tagdir " << tagdir 
                   << " basedir " << ( basedir ? basedir : "-" )
                   << " incl_path " << ( incl_path ? incl_path : "-" )
                   ; 
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
    LOG(debug) << "Prog::examine tagdir [" << tagdir << "]"  ; 

    for(unsigned int i=0 ; i < m_names.size() ; ++i)
    {
        std::string glsl = BFile::FormPath(tagdir, m_names[i].c_str()) ;  
        if(BFile::ExistsNativeFile(glsl))
        {
            const char* path = glsl.c_str();
            GLenum type = m_codes[i] ;
            Shdr* shdr = new Shdr(path, type, m_incl_path.c_str()); 
            m_shaders.push_back(shdr);
        }
        else
        {
            LOG(debug) << "Prog::examine didnt find glsl file " << glsl ; 
        } 
    }     
}

void Prog::createOnly()
{
    if(m_verbosity > 1) LOG(info) << "Prog::createOnly" << desc() ;
 
    create();
}

void Prog::linkAndValidate()
{
    if(m_verbosity > 1) LOG(info) << "Prog::linkAndValidate" << desc()  ; 

    link();
    validate();
    collectLocations();
}

void Prog::createAndLink()
{
    if(m_verbosity > 1) LOG(info) << "Prog::createAndlink" << desc()  ; 
    G::ErrCheck("Prog::createAndLink.[", true);

    create();
    link();
    validate();
    collectLocations();
    //Print("Prog::createAndLink");
    G::ErrCheck("Prog::createAndLink.]", true);
}


void Prog::create()
{
    G::ErrCheck("Prog::create.[", true);
    m_id = glCreateProgram();
    G::ErrCheck("Prog::create.m_id", true);

    for(unsigned int i=0 ; i<m_shaders.size() ; i++)
    {
        G::ErrCheck("Prog::create.loop.0", true);
        Shdr* shdr = m_shaders[i];
        shdr->createAndCompile();           
        glAttachShader (m_id, shdr->getId());
        G::ErrCheck("Prog::create.loop.1", true);
    }

    G::ErrCheck("Prog::create.]", true);
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
    if(m_verbosity > 1) LOG(info) << "Prog::validate" << desc()  ; 

    glValidateProgram (m_id);

    int params = -1;
    glGetProgramiv (m_id, GL_VALIDATE_STATUS, &params);

    if (GL_TRUE != params) 
    {

        ProgLog prl(m_id) ;

        if(m_nofrag && prl.is_no_frag_shader() )
        {
            if(m_verbosity > 1)
            LOG(info) << "ignoring lack of frag shader "  ; 
        } 
        else
        {
            _print_program_info_log();
            prl.dump("Prog::validate");
            Print("Prog::validate ERROR"); 
            LOG(fatal) << "Prog::validate failure for tagdir [" << m_tagdir << "] EXITING "  ; 
            exit(1);
        }
    }
}

void Prog::collectLocations()
{
    bool print = false ; 
    traverseActive(Attribute, print);

    if(!m_ubo)
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
    const char* note = "\n NB remember you must use the uniform in the shader otherwise it gets optimized away and WILL NOT BE FOUND " ; 
    std::string name(name_);
    if(m_uniforms.find(name) == m_uniforms.end())
    {
         if(required)
         {
             LOG(fatal) << "Prog::uniform " << m_tagdir << " failed to find required uniform [" << name << "]" 
                        << note ;
             dumpUniforms(); 
             exit(1);
         }
         else
         {
             LOG(debug) << "Prog::uniform " << m_tagdir << " did not find optional uniform [" << name << "]" 
                        << note ; 
         }
         return -1 ; 
    }
    return m_uniforms[name];
}





void Prog::_print_program_info_log()
{
    LOG(info) << "Prog::_print_program_info_log" << desc() ; 

    ProgLog prl(m_id) ;
    prl.dump("Prog::_print_program_info_log");

    LOG(info) << " NO_FRAGMENT_SHADER " << prl.is_no_frag_shader() ;  

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


/**
Prog::traverseActive
----------------------

Introspect the shader collecting active Uniforms or Attributes into 
m_uniforms or m_attributes.

**/


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
        GLenum type(0);

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


