// http://antongerdelan.net/opengl/shaders.html

#include <GL/glew.h>
#include "Shader.hh"
#include "stdlib.h"
#include "stdio.h"
#include "assert.h"

const char* Shader::vertex_shader =
"#version 400\n"
"in vec3 vp;"
"void main () {"
"  gl_Position = vec4 (vp, 1.0);"
"}";

const char* Shader::fragment_shader =
"#version 400\n"
"out vec4 frag_colour;"
"void main () {"
"  frag_colour = vec4 (0.5, 0.0, 0.5, 1.0);"
"}";


Shader::Shader()
{
   init(); 
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
  printf ("--------------------\nshader programme %i info:\n", programme);
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
  printf ("program %i GL_VALIDATE_STATUS = %i\n", programme, params);
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

void Shader::init()
{
    m_vs = glCreateShader (GL_VERTEX_SHADER);
    glShaderSource (m_vs, 1, &vertex_shader, NULL);
    compile(m_vs);

    m_fs = glCreateShader (GL_FRAGMENT_SHADER);
    glShaderSource (m_fs, 1, &fragment_shader, NULL);
    compile(m_fs);
    
    m_program = glCreateProgram ();
    glAttachShader (m_program, m_fs);
    glAttachShader (m_program, m_vs);
    link(m_program);

    assert( isValid() );
} 

void Shader::dump()
{
    print_all(m_program);
}

bool Shader::isValid()
{
    return _is_valid(m_program);
}


GLuint Shader::getProgram()
{
    return m_program ; 
}

