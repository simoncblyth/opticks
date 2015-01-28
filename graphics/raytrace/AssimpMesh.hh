#ifndef ASSIMPMESH_H
#define ASSIMPMESH_H


struct aiMesh ; 
#include <assimp/types.h>

class AssimpMesh  {

public:
    AssimpMesh(aiMesh* mesh, const aiMatrix4x4& transform);

    virtual ~AssimpMesh();

private:
    void copy(aiMesh* mesh, const aiMatrix4x4& transform);

public:
    aiMesh* getRawMesh();

private:

    aiMesh* m_mesh ; 


};


#endif
