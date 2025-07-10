#pragma once
/**
SGLFW_VAO : Minimal Vertex Array wrapper
--------------------------------------------


**/

struct SGLFW_VAO
{
    const char* name ;
    GLuint id ;

    SGLFW_VAO(const char* name);
    void init();
    void bind();
    void unbind();
};

inline SGLFW_VAO::SGLFW_VAO(const char* _name)
    :
    name(_name ? strdup(_name) : nullptr),
    id(-1)
{
    init();
}

inline void SGLFW_VAO::init()
{
    glGenVertexArrays (1, &id);  SGLFW__check(__FILE__, __LINE__, name, id, "glGenVertexArrays/init" );
}

inline void SGLFW_VAO::bind()
{
    glBindVertexArray(id);        SGLFW__check(__FILE__, __LINE__, name, id, "glBindVertexArray/bind" );
}
inline void SGLFW_VAO::unbind()
{
    glBindVertexArray(0);        SGLFW__check(__FILE__, __LINE__, name, id, "glBindVertexArray/unbi" );
}



