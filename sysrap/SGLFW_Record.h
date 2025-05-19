#pragma once
/**
SGLFW_Record.h
===============


**/


struct SGLFW_Program ;
struct SRecordInfo ;

struct SGLFW_Record
{
    static SGLFW_Record* Create(const SRecordInfo* _record, const float* _timeparam_ptr  );

    const SRecordInfo*  record ;
    const float*        timeparam_ptr ;

    SGLFW_Buffer*       rpos ;
    SGLFW_VAO*          vao ;
    GLint               param_location;

    SGLFW_Record(const SRecordInfo* _record, const float* _timeparam_ptr  ) ;
    void init();

    void render(const SGLFW_Program* prog);
};


inline SGLFW_Record* SGLFW_Record::Create(const SRecordInfo* _record, const float* _timeparam_ptr )
{
    return _record ? new SGLFW_Record(_record, _timeparam_ptr) : nullptr ;
}


inline SGLFW_Record::SGLFW_Record(const SRecordInfo* _record, const float* _timeparam_ptr )
    :
    record(_record),
    timeparam_ptr(_timeparam_ptr),
    rpos(nullptr),
    vao(nullptr),
    param_location(0)
{
    init();
}

/**
SGLFW_Record::init
----------------


**/


inline void SGLFW_Record::init()
{
    vao = new SGLFW_VAO ;  // vao: establishes context for OpenGL attrib state and element array (not GL_ARRAY_BUFFER)
    vao->bind();

    rpos = new SGLFW_Buffer( record->record->arr_bytes(), record->record->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW );
    rpos->bind();
    rpos->upload();
}


/**
SGLFW_Record::render
---------------------

Called from renderloop.

**/

inline void SGLFW_Record::render(const SGLFW_Program* prog)
{
    param_location = prog->getUniformLocation("Param");
    prog->use();
    vao->bind();

    rpos->bind();
    prog->enableVertexAttribArray("rpos", SRecordInfo::RPOS_SPEC );

    if(param_location > -1 ) prog->Uniform4fv(param_location, timeparam_ptr, false );
    prog->updateMVP();  // ?

    GLenum mode = prog->geometry_shader_text ? GL_LINE_STRIP : GL_POINTS ;
    glDrawArrays(mode, record->record_first,  record->record_count);
}


