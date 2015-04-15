// http://antongerdelan.net/opengl/shaders.html

#include <GL/glew.h>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include "Shader.hh"
#include "stdlib.h"
#include "stdio.h"
#include "assert.h"

#include <string>
#include <iostream>
#include <fstream>

const char* Shader::vertex_shader =
"#version 400\n"
"uniform mat4 ModelView;"
"uniform mat4 ModelViewProjection;"
"layout(location = 0) in vec3 vertex_position;"
"layout(location = 1) in vec3 vertex_colour;"
"layout(location = 2) in vec3 vertex_normal;"
"layout(location = 3) in vec2 vertex_texcoord;"
"out vec3 colour;"
"out vec2 texcoord;"
"void main () {"
"  vec4 normal = ModelView * vec4 (vertex_normal, 0.0);"
"  colour = normalize(vec3(normal))*0.5 + 0.5 ;"
"  gl_Position = ModelViewProjection * vec4 (vertex_position, 1.0);"
"  texcoord = vertex_texcoord;"
"}";

const char* Shader::fragment_shader =
"#version 400\n"
"in vec3 colour;"
"in vec2 texcoord;"
"out vec4 frag_colour;"
"uniform sampler2D texSampler;"
"void main () {"
"  //frag_colour = texture(texSampler, texcoord);" 
"  frag_colour = vec4 (colour, 1.0);"
"}";



std::string readFile(const char *path) 
{
    //std::cout << "readFile " << path << std::endl ; 
    LOG(debug) << "readFile " << path << std::endl ; 

    std::string content;
    std::ifstream fs(path, std::ios::in);
    if(!fs.is_open()) {
        std::cerr << "readFile FAILED for : " << std::endl;
        return "";
    }

    std::string line = "";
    while(!fs.eof()) {
        std::getline(fs, line);
        content.append(line + "\n");
    }

    fs.close();
    return content;
}



Shader::Shader(const char* basedir, const char* tag, const char* vname, const char* fname)
{
   std::string vert ;
   std::string frag ;

   if(basedir)
   {
       char vpath[256];
       char fpath[256];
       snprintf(vpath, 256, "%s/%s/%s", basedir, tag, vname);
       snprintf(fpath, 256, "%s/%s/%s", basedir, tag, fname);
       vert = readFile(vpath) ;
       frag = readFile(fpath) ;
       LOG(info) << "Shader::Shader vpath " << vpath;
       LOG(info) << "Shader::Shader fpath " << fpath;
   } 
   else
   {
       LOG(warning) << "Shader::Shader WARNING using default shaders ";
       vert = vertex_shader ; 
       frag = fragment_shader ; 
   }

   init(vert, frag); 
}
Shader::~Shader()
{
}


void _print_shader_info_log (GLuint index) {
  int max_length = 2048;
  int actual_length = 0;
  char log[2048];
  glGetShaderInfoLog (index, max_length, &actual_length, log);
  printf ("shader info log for GL index %u:\n%s\n", index, log);
}

void _print_program_info_log (GLuint index) {
  int max_length = 2048;
  int actual_length = 0;
  char log[2048];
  glGetProgramInfoLog (index, max_length, &actual_length, log);
  printf ("program info log for GL index %u:\n%s", index, log);
}



const char* GL_type_to_string (GLenum type) {
  switch (type) {
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


void print_all (GLuint programme) {
  printf ("Shader:print_all -----------------\nshader programme %i info:\n", programme);
  int params = -1;
  glGetProgramiv (programme, GL_LINK_STATUS, &params);
  printf ("GL_LINK_STATUS = %i\n", params);
  
  glGetProgramiv (programme, GL_ATTACHED_SHADERS, &params);
  printf ("GL_ATTACHED_SHADERS = %i\n", params);
  
  glGetProgramiv (programme, GL_ACTIVE_ATTRIBUTES, &params);
  printf ("GL_ACTIVE_ATTRIBUTES = %i\n", params);
  for (int i = 0; i < params; i++) {
    char name[64];
    int max_length = 64;
    int actual_length = 0;
    int size = 0;
    GLenum type;
    glGetActiveAttrib (
      programme,
      i,
      max_length,
      &actual_length,
      &size,
      &type,
      name
    );
    if (size > 1) {
      for (int j = 0; j < size; j++) {
        char long_name[64];
        sprintf (long_name, "%s[%i]", name, j);
        int location = glGetAttribLocation (programme, long_name);
        printf ("  %i) type:%s name:%s location:%i\n",
          i, GL_type_to_string (type), long_name, location);
      }
    } else {
      int location = glGetAttribLocation (programme, name);
      printf ("  %i) type:%s name:%s location:%i\n",
        i, GL_type_to_string (type), name, location);
    }
  }
  
  glGetProgramiv (programme, GL_ACTIVE_UNIFORMS, &params);
  printf ("GL_ACTIVE_UNIFORMS = %i\n", params);
  for (int i = 0; i < params; i++) {
    char name[64];
    int max_length = 64;
    int actual_length = 0;
    int size = 0;
    GLenum type;
    glGetActiveUniform (
      programme,
      i,
      max_length,
      &actual_length,
      &size,
      &type,
      name
    );
    if (size > 1) {
      for (int j = 0; j < size; j++) {
        char long_name[64];
        sprintf (long_name, "%s[%i]", name, j);
        int location = glGetUniformLocation (programme, long_name);
        printf ("  %i) type:%s name:%s location:%i\n",
          i, GL_type_to_string (type), long_name, location);
      }
    } else {
      int location = glGetUniformLocation (programme, name);
      printf ("  %i) type:%s name:%s location:%i\n",
        i, GL_type_to_string (type), name, location);
    }
  }
  
  _print_program_info_log (programme);
}



bool _is_valid(GLuint programme) {
  glValidateProgram (programme);
  int params = -1;
  glGetProgramiv (programme, GL_VALIDATE_STATUS, &params);
  //printf ("program %i GL_VALIDATE_STATUS = %i\n", programme, params);
  if (GL_TRUE != params) {
    _print_program_info_log (programme);
    return false;
  }
  return true;
}


void Shader::compile(GLuint index)
{
    glCompileShader (index);
    int params = -1;
    glGetShaderiv (index, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) {
        fprintf (stderr, "ERROR: GL shader index %i did not compile\n", index);
        _print_shader_info_log (index);
        exit(1); 
    } 
}

void Shader::link(GLuint index)
{
    glLinkProgram (index);
    int params = -1;
    glGetProgramiv (index, GL_LINK_STATUS, &params);
    if (GL_TRUE != params) {
        fprintf (stderr, "ERROR: linking GL shader program index %i \n", index);
        _print_program_info_log (index);
        exit(1); 
    } 
}

void Shader::init(const std::string& vert, const std::string& frag)
{
    m_vs = glCreateShader (GL_VERTEX_SHADER);

    const char* vert_c = vert.c_str();
    glShaderSource (m_vs, 1, &vert_c, NULL);
    compile(m_vs);

    const char* frag_c = frag.c_str();
    m_fs = glCreateShader (GL_FRAGMENT_SHADER);
    glShaderSource (m_fs, 1, &frag_c, NULL);
    compile(m_fs);
    
    m_program = glCreateProgram ();
    glAttachShader (m_program, m_fs);
    glAttachShader (m_program, m_vs);
    link(m_program);
    
    m_mvp_location = glGetUniformLocation(m_program, "ModelViewProjection");
    m_mv_location = glGetUniformLocation(m_program, "ModelView");
    m_sampler_location = glGetUniformLocation(m_program, "texSampler");

    //Print("init");

    assert( isValid() );
} 

void Shader::use()
{
}


void Shader::Print(const char* msg)
{
    printf("Shader::%s locations mvp/mv/sampler : %d/%d/%d ", msg, m_mvp_location, m_mv_location, m_sampler_location );
}



GLint Shader::getMVPLocation()
{
    return m_mvp_location ; 
}
GLint Shader::getMVLocation()
{
    return m_mv_location ; 
}
GLint Shader::getSamplerLocation()
{
    return m_sampler_location ; 
}




void Shader::dump(const char* msg)
{
    printf("%s Shader::dump\n", msg);
    printf("m_mvp_location %d \n", m_mvp_location );
    printf("m_mv_location %d \n", m_mv_location );
    print_all(m_program);
}

bool Shader::isValid()
{
    return _is_valid(m_program);
}

GLuint Shader::getId()
{
    return m_program ; 
}

