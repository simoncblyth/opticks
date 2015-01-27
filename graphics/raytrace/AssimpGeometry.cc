#include "AssimpGeometry.hh"

#include <string.h>
#include <stdlib.h>
#include <sstream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>


static unsigned int findNode_index = 0 ; 

aiNode* findNode(const char* query, aiNode* node, unsigned int depth )
{
   if(depth == 0) findNode_index = 0 ; 

   //dumpNode(node, depth); 

   findNode_index++ ; 

   const char* name = node->mName.C_Str(); 

   if(strncmp(name,query,strlen(query)) == 0) return node;

   for(unsigned int i = 0; i < node->mNumChildren; i++)
   {   
      aiNode* n = findNode(query, node->mChildren[i], depth + 1 );
      if(n) return n ; 
   }   
   return NULL ; 
}


void dumpNode(aiNode* node, unsigned int depth)
{
   if(!node)
   {   
      printf("dumpNode NULL \n");
      return ; 
   }   

   unsigned int NumMeshes = node->mNumMeshes ;
   unsigned int NumChildren = node->mNumChildren ;
   const char* name = node->mName.C_Str(); 
   printf("i %5d d %2d m %3d c %3d n %s \n", findNode_index, depth, NumMeshes, NumChildren, name); 

   /*  
   if(findNode_index > 0 )
   {
       // other than first, even node index have 1 mesh and odd have 0
       assert( (findNode_index + 1) % 2 == NumMeshes );

       // the odd zeros, always have children 
       if(NumMeshes == 0) assert(NumChildren > 0 ); 
   }
   */
}


void dumpTree(aiNode* node, unsigned int depth)
{

   if(!node)
   {
      printf("dumpTree NULL \n");
      return ;
   }

   if(depth == 0) findNode_index = 0 ;

   dumpNode(node, depth);

   findNode_index++ ;

   for(unsigned int i = 0; i < node->mNumChildren; i++)
   {
       dumpTree(node->mChildren[i], depth + 1);
   }
}



AssimpGeometry::AssimpGeometry(const char* path, const char* query )
          : 
          m_path(NULL),
          m_query(NULL),
          m_importer(new Assimp::Importer()),
          m_aiscene(NULL)
{
    if(!path || !query) return ;          

    printf("AssimpGeometry::AssimpGeometry ctor path %s query %s  \n", path, query );
    m_path = strdup(path);
    m_query = strdup(query);
}

AssimpGeometry::~AssimpGeometry()
{
    // deleting m_importer also deletes the scene (unconfirmed)
    delete m_importer;
    free(m_path);
    free(m_query);
}


void AssimpGeometry::info()
{
    printf("AssimpGeometry::info aiscene %p \n", m_aiscene);
    if(!m_aiscene) return ; 
    printf("aiscene Flags         %d \n", m_aiscene->mFlags );
    printf("aiscene NumAnimations %d \n", m_aiscene->mNumAnimations );
    printf("aiscene NumCameras    %d \n", m_aiscene->mNumCameras );
    printf("aiscene NumLights     %d \n", m_aiscene->mNumLights );
    printf("aiscene NumMaterials  %d \n", m_aiscene->mNumMaterials );
    printf("aiscene NumMeshes     %d \n", m_aiscene->mNumMeshes );
    printf("aiscene NumTextures   %d \n", m_aiscene->mNumTextures );
}



void AssimpGeometry::import()
{

    m_aiscene = m_importer->ReadFile( m_path, 
                     aiProcess_CalcTangentSpace       |   
                     aiProcess_Triangulate            |   
                     aiProcess_JoinIdenticalVertices  |
                     aiProcess_SortByPType);

    if(!m_aiscene)
    {   
        printf("import error : %s \n", m_importer->GetErrorString() );  
    }   

    info();
}


aiNode* AssimpGeometry::searchNode(const char* query)
{
   aiNode* root = m_aiscene ? m_aiscene->mRootNode : NULL ;
   if(!root)
   {
       printf("rootnode not defined \n");
       return NULL ; 
   }
   aiNode* node = findNode(query, root, 0); 

   dumpTree(node, 0 );

   return node ; 
}


void AssimpGeometry::dumpMaterial(aiMaterial* ai_material)
{
    aiString name;
    ai_material->Get(AI_MATKEY_NAME, name);
    unsigned int numProperties = ai_material->mNumProperties ;
    printf("Assimp::dumpMaterial props %2d %s \n", numProperties, name.C_Str());

    for(unsigned int i = 0; i < ai_material->mNumProperties; i++)
    {
        aiMaterialProperty* property = ai_material->mProperties[i] ;
        aiString key = property->mKey ; 
        printf("key %s \n", key.C_Str());
    }
}



