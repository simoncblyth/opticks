#pragma once

struct SMesh ; 

/**
SGLFW_Render
-------------

Try same prog for multiple mesh 

**/
struct SGLFW_Render
{
    const SMesh*   mesh ;  
    SGLFW_Program* prog ;  

    SGLFW_Buffer*  vtx ; 
    SGLFW_Buffer*  nrm ;

    SGLFW_VAO*     vao ;  
    SGLFW_Buffer*  idx ; 

    SGLFW_Render(const SMesh* mesh, SGLFW_Program* prog ) ; 
    void render(); 

};

inline SGLFW_Render::SGLFW_Render(const SMesh* _mesh, SGLFW_Program* _prog )
    :
    mesh(_mesh),
    prog(_prog),
    vtx(nullptr),
    nrm(nullptr),
    vao(nullptr),
    idx(nullptr)
{
    vtx = new SGLFW_Buffer( mesh->vtx->arr_bytes(), mesh->vtx->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    vtx->bind();
    vtx->upload(); 

    nrm = new SGLFW_Buffer( mesh->nrm->arr_bytes(), mesh->nrm->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    nrm->bind();
    nrm->upload(); 

    vao = new SGLFW_VAO ;  // vao: establishes context for OpenGL attrib state and element array (not vbuf,nbuf)
    vao->bind(); 

    idx = new SGLFW_Buffer( mesh->tri->arr_bytes(), mesh->tri->cvalues<int>()  , GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW ); 
    idx->bind();
    idx->upload(); 
}

inline void SGLFW_Render::render()
{
   prog->use(); 
   vao->bind(); 

   vtx->bind();
   prog->enableVertexAttribArray( prog->vtx_attname, mesh->vtx_spec ); 

   nrm->bind();
   prog->enableVertexAttribArray( prog->nrm_attname, mesh->nrm_spec ); 

   // NB: careful with the ordering of the above or the OpenGL state machine will bite you : 
   // the vPos and vNrm attribs needs to be enabled after the appropriate buffer is made THE active GL_ARRAY_BUFFER

   idx->bind();

   prog->updateMVP();

   glDrawElements(GL_TRIANGLES, mesh->indices_num, GL_UNSIGNED_INT, (GLvoid*)(sizeof(GLuint) * mesh->indices_offset ));
}



