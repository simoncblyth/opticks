#include "AssimpMesh.hh"

#include <assimp/scene.h>


AssimpMesh::AssimpMesh(aiMesh* mesh, const aiMatrix4x4& mat ) : m_mesh(NULL)
{
   m_mesh = new aiMesh ; 
   copy(mesh, mat);
}

AssimpMesh::~AssimpMesh()
{
   delete m_mesh ; 
}

void AssimpMesh::copy(aiMesh* mesh, const aiMatrix4x4& mat )
{
   //  /usr/local/env/graphics/assimp/assimp-3.1.1/code/PretransformVertices.cpp
   //  /usr/local/env/graphics/assimp/assimp-3.1.1/code/ColladaLoader.cpp

   m_mesh->mPrimitiveTypes = mesh->mPrimitiveTypes ;

   if (mesh->HasPositions()) 
   {
       m_mesh->mNumVertices    = mesh->mNumVertices ;
       m_mesh->mVertices = new aiVector3D[mesh->mNumVertices];
       for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
            m_mesh->mVertices[i] = mat * mesh->mVertices[i];
       }   
   }   
  

   m_mesh->mNumFaces = mesh->mNumFaces ;
   m_mesh->mFaces = new aiFace[mesh->mNumFaces];

   for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
      m_mesh->mFaces[i] = mesh->mFaces[i] ;
   }

}


aiMesh* AssimpMesh::getMesh()
{
   return m_mesh ;  
}


