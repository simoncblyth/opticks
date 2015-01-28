#include "AssimpCommon.hh"

#include <assimp/scene.h>
#include <assimp/material.h>


aiNode* findNode(const char* query, aiNode* node, unsigned int depth ){
   const char* name = node->mName.C_Str(); 
   if(strncmp(name,query,strlen(query)) == 0) return node;
   for(unsigned int i = 0; i < node->mNumChildren; i++){   
      aiNode* n = findNode(query, node->mChildren[i], depth + 1 );
      if(n) return n ; 
   }   
   return NULL ; 
}

void dumpNode(aiNode* node, unsigned int depth){
   if(!node) return ;
   unsigned int NumMeshes = node->mNumMeshes ;
   unsigned int NumChildren = node->mNumChildren ;
   const char* name = node->mName.C_Str(); 
   printf("d %2d m %3d c %3d n %s \n", depth, NumMeshes, NumChildren, name); 
}

void dumpMesh( aiMesh* mesh ){
    unsigned int numFaces = mesh->mNumFaces;
    unsigned int numVertices = mesh->mNumVertices;
    for(unsigned int i=0 ; i < numVertices ; i++ )
    {
        aiVector3D& v = mesh->mVertices[i] ;
        if( i == 0 || i == numVertices - 1) 
        printf("i %4d  xyz %10.3f %10.3f %10.3f \n", i, v.x, v.y, v.z ); 
    }
}

void dumpMaterial( aiMaterial* material )
{
    aiString name;
    material->Get(AI_MATKEY_NAME, name);
    unsigned int numProperties = material->mNumProperties ;
    for(unsigned int i = 0; i < material->mNumProperties; i++)
    {
        aiMaterialProperty* property = material->mProperties[i] ;
        aiString key = property->mKey ; 
        printf("key %s \n", key.C_Str());
    }
    printf("dumpMaterial props %2d %s \n", numProperties, name.C_Str());
}

void dumpTransform(aiMatrix4x4 t)
{
   printf("dumpTransform\n");
   printf("a %10.4f %10.4f %10.4f %10.4f \n", t.a1, t.a2, t.a3, t.a4 );
   printf("b %10.4f %10.4f %10.4f %10.4f \n", t.b1, t.b2, t.b3, t.b4 );
   printf("c %10.4f %10.4f %10.4f %10.4f \n", t.c1, t.c2, t.c3, t.c4 );
   printf("d %10.4f %10.4f %10.4f %10.4f \n", t.d1, t.d2, t.d3, t.d4 );
}







