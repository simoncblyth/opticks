#pragma once
/**
SGLDisplay.h : OpenGL shader pipeline that presents PBO to screen
==================================================================


**/

#include <cstdio>
#include <cstdint>
#include <string>
#include <iostream>

struct SGLDisplay
{
    enum BufferImageFormat
    {
        UNSIGNED_BYTE4,
        FLOAT4,
        FLOAT3
    };

    static GLuint CreateGLShader( const char* source, GLuint shader_type );
    static GLuint CreateGLProgram( const char* vert_source, const char* frag_source ); 
    static size_t PixelFormatSize( BufferImageFormat format ); 
    static int PixelUnpackAlignment( size_t elmt_size ); 

    SGLDisplay(BufferImageFormat format = UNSIGNED_BYTE4);
    void init(); 
    std::string desc() const ; 

    void display(
            const int32_t  screen_res_x,
            const int32_t  screen_res_y,
            const int32_t  framebuf_res_x,
            const int32_t  framebuf_res_y,
            const uint32_t pbo,
            bool dump = false 
           );

    BufferImageFormat m_image_format;
    size_t  m_elmt_size ; 
    int     m_unpack_alignment ; 
    size_t  m_count ; 

    GLuint   m_render_tex = 0u;
    GLuint   m_program = 0u;
    GLint    m_render_tex_uniform_loc = -1;
    GLuint   m_quad_vertex_buffer = 0;

    static constexpr const char* s_vert_source = R"(
#version 330 core

// transforms ndc (-1:1,-1:1) "vertexPosition_modelspace"  to tex coords (0:1,0:1 ) "UV"

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
    gl_Position =  vec4(vertexPosition_modelspace,1);
    UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

    static constexpr const char* s_frag_source = R"(
#version 330 core

// samples texture at UV coordinate

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";

};

/**
SGLDisplay::CreateGLShader
---------------------------

Compile shader from provided source string

**/

inline GLuint SGLDisplay::CreateGLShader( const char* source, GLuint shader_type ) // static
{
    GLuint shader = glCreateShader( shader_type );
    {
        const GLchar* source_data= reinterpret_cast<const GLchar*>( source );
        glShaderSource( shader, 1, &source_data, nullptr );
        glCompileShader( shader );

        GLint is_compiled = 0;
        glGetShaderiv( shader, GL_COMPILE_STATUS, &is_compiled );
        if( is_compiled == GL_FALSE )
        {
            GLint max_length = 0;
            glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &max_length );
            std::string info_log( max_length, '\0' );
            GLchar* info_log_data= reinterpret_cast<GLchar*>( &info_log[0]);
            glGetShaderInfoLog( shader, max_length, nullptr, info_log_data );

            glDeleteShader(shader);
            std::cerr << "Compilation of shader failed: " << info_log << std::endl;

            return 0;
        }
        GL_CHECK_ERRORS();
    } 
    return shader;
}

/**
SGLDisplay::CreateGLProgram
-----------------------------

Create pipeline from vertex and fragment shader sources

**/


inline GLuint SGLDisplay::CreateGLProgram( const char* vert_source, const char* frag_source ) // static
{
    GLuint vert_shader = CreateGLShader( vert_source, GL_VERTEX_SHADER );
    if(vert_shader == 0) return 0;

    GLuint frag_shader = CreateGLShader( frag_source, GL_FRAGMENT_SHADER );
    if(frag_shader == 0)
    {
        glDeleteShader( vert_shader );
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader( program, vert_shader );
    glAttachShader( program, frag_shader );
    glLinkProgram( program );

    GLint is_linked = 0;
    glGetProgramiv( program, GL_LINK_STATUS, &is_linked );
    if (is_linked == GL_FALSE)
    {
        GLint max_length = 0;
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &max_length );

        std::string info_log( max_length, '\0' );
        GLchar* info_log_data= reinterpret_cast<GLchar*>( &info_log[0]);
        glGetProgramInfoLog( program, max_length, nullptr, info_log_data );
        std::cerr << "Linking of program failed: " << info_log << std::endl;

        glDeleteProgram( program );
        glDeleteShader( vert_shader );
        glDeleteShader( frag_shader );

        return 0;
    }

    glDetachShader( program, vert_shader );
    glDetachShader( program, frag_shader );

    GL_CHECK_ERRORS();

    return program;
}


inline size_t SGLDisplay::PixelFormatSize( BufferImageFormat format ) // static
{
    switch( format )
    {    
        case UNSIGNED_BYTE4: return sizeof( char ) * 4; 
        case FLOAT3:         return sizeof( float ) * 3; 
        case FLOAT4:         return sizeof( float ) * 4; 
        default:
            throw std::runtime_error( "SGLDisplay::PixelFormatSize: Unrecognized buffer format" );
    }    
}

inline int SGLDisplay::PixelUnpackAlignment( size_t elmt_size ) // static
{
    int unpack_alignment = 1 ; 
    if      ( elmt_size % 8 == 0) unpack_alignment = 8 ; 
    else if ( elmt_size % 4 == 0) unpack_alignment = 4 ; 
    else if ( elmt_size % 2 == 0) unpack_alignment = 2 ; 
    else                          unpack_alignment = 1 ; 
    return unpack_alignment ; 
}




/**
SGLDisplay::SGLDisplay
-----------------------

**/

inline SGLDisplay::SGLDisplay( BufferImageFormat image_format )
    : 
    m_image_format(image_format),
    m_elmt_size(PixelFormatSize( m_image_format )),
    m_unpack_alignment(PixelUnpackAlignment( m_elmt_size )),
    m_count(0)
{
    init(); 
}


/**
SGLDisplay::init
-----------------

1. binds Vertex Array m_vertex_array 
2. creates shader pipeline m_program  
3. binds and configures m_render_tex GL_TEXTURE_2D
4. binds m_quad_vertex_buffer GL_ARRAY_BUFFER 
   and uploads vertices of two triangles forming a [-1:1,-1:1] quad

**/

inline void SGLDisplay::init()
{
    GLuint m_vertex_array;
    GL_CHECK( glGenVertexArrays(1, &m_vertex_array ) );
    GL_CHECK( glBindVertexArray( m_vertex_array ) );

    m_program = CreateGLProgram( s_vert_source, s_frag_source );
    m_render_tex_uniform_loc = glGetUniformLocation( m_program, "render_tex" );
    assert( m_render_tex_uniform_loc > -1 ); 

    GL_CHECK( glGenTextures( 1, &m_render_tex ) );
    GL_CHECK( glBindTexture( GL_TEXTURE_2D, m_render_tex ) );

    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ) );
    GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE ) );

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
    };

    GL_CHECK( glGenBuffers( 1, &m_quad_vertex_buffer ) );
    GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_quad_vertex_buffer ) );
    GL_CHECK( glBufferData( GL_ARRAY_BUFFER,
                sizeof( g_quad_vertex_buffer_data),
                g_quad_vertex_buffer_data,
                GL_STATIC_DRAW
                )
            );

    GL_CHECK_ERRORS();
}

inline std::string SGLDisplay::desc() const
{
    std::stringstream ss ; 
    ss << "SGLDisplay::desc"
       << std::endl 
       << " int(m_image_format) " << int(m_image_format)
       << std::endl 
       << " m_elmt_size " << m_elmt_size 
       << std::endl 
       << " m_unpack_alignment " << m_unpack_alignment
       << std::endl 
       << " m_count " << m_count
       << std::endl 
       << " m_render_tex " << m_render_tex
       << std::endl 
       << " m_program "
       << std::endl 
       << " m_render_tex_uniform_loc " << m_render_tex_uniform_loc
       << std::endl 
       << " m_quad_vertex_buffer " << m_quad_vertex_buffer
       << std::endl 
       ;
    std::string str = ss.str(); 
    return str ; 
}

/**
SGLDisplay::display
--------------------

Arranges for the PBO buffer contents to be viewed as a texture
and accessed via samplers in shaders.

1. bind GL_FRAMEBUFFER to 0 (break any bind, return to screen rendering)
2. set viewport to (framebuf_res_x, framebuf_res_y) 
3. bind GL_TEXTURE_2D m_render_tex
4. configure GL_PIXEL_UNPACK_BUFFER unpacking 
5. invoke glTexImage2D using (screen_res_x, screen_res_y) 
   which enables shaders to access the texture

   * note that nullptr data, because GL_PIXEL_UNPACK_BUFFER
     is bound the pbo acts as the source of the texture data 
     and the data arg is just an offset 

6. configure vertex array access from shader
7. draw the two triangles of the quad displaying PBO data
   via viewing it as a texture 

 
glTexImage2D 
   specifies mutable texture storage characteristics and provides the data
  
   *internalFormat* 
        format with which OpenGL should store the texels in the texture

   *data*
        location of the initial texel data in host memory, 
        **if a buffer is bound to the GL_PIXEL_UNPACK_BUFFER binding point, 
        texel data is read from that buffer object, and *data* is interpreted 
        as an offset into that buffer object from which to read the data**

    *format* and *type*
         initial source texel data layout which OpenGL will convert 
         to the internalFormat

**/

inline void SGLDisplay::display(
        const int32_t  screen_res_x,
        const int32_t  screen_res_y,
        const int32_t  framebuf_res_x,
        const int32_t  framebuf_res_y,
        const uint32_t pbo,
        bool dump
        ) 
{
    if(dump) printf("[ SGLDisplay::display count %zu framebuf_res_x %d framebuf_res_y %d  \n", m_count,framebuf_res_x, framebuf_res_y  ); 

    GL_CHECK( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) ); 
    // "FBO" name:0 means break the bind : so that typically 
    // means switch to default rendering to the screen 
    // https://learnopengl.com/Advanced-OpenGL/Framebuffers

    GL_CHECK( glViewport( 0, 0, framebuf_res_x, framebuf_res_y ) );
    GL_CHECK( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) );
    GL_CHECK( glUseProgram( m_program ) );

    // Bind our texture in Texture Unit 0
    GL_CHECK( glActiveTexture( GL_TEXTURE0 ) );
    GL_CHECK( glBindTexture( GL_TEXTURE_2D, m_render_tex ) );
    GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo ) );
    GL_CHECK( glPixelStorei(GL_UNPACK_ALIGNMENT, m_unpack_alignment ) ); 

    bool convertToSrgb = true;

    if( m_image_format == BufferImageFormat::UNSIGNED_BYTE4 )
    {
        // input is assumed to be in srgb since it is only 1 byte per channel in size
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8,   screen_res_x, screen_res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr );
        convertToSrgb = false;
    }
    else if( m_image_format == BufferImageFormat::FLOAT3 )
    {
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F,  screen_res_x, screen_res_y, 0, GL_RGB,  GL_FLOAT,         nullptr );
    }
    else if( m_image_format == BufferImageFormat::FLOAT4 )
    {
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, screen_res_x, screen_res_y, 0, GL_RGBA, GL_FLOAT,         nullptr );
    }
    else
    {
        throw std::runtime_error( "Unknown buffer format" );
    }

    GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 ) );
    GL_CHECK( glUniform1i( m_render_tex_uniform_loc , 0 ) );

    // 1st attribute buffer : vertices
    GL_CHECK( glEnableVertexAttribArray( 0 ) );
    GL_CHECK( glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer ) );
    GL_CHECK( glVertexAttribPointer(
            0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
            )
        );

    if( convertToSrgb )
        GL_CHECK( glEnable( GL_FRAMEBUFFER_SRGB ) );
    else 
        GL_CHECK( glDisable( GL_FRAMEBUFFER_SRGB ) );

    // Draw the triangles !
    GL_CHECK( glDrawArrays(GL_TRIANGLES, 0, 6) ); // 2*3 indices starting at 0 -> 2 triangles

    GL_CHECK( glDisableVertexAttribArray(0) );

    GL_CHECK( glDisable( GL_FRAMEBUFFER_SRGB ) );

    if(dump) printf("] SGLDisplay::display %zu \n", m_count ); 
    m_count++ ; 
    GL_CHECK_ERRORS();
}

