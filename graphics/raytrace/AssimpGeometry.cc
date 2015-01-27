#include "AssimpGeometry.hh"

/*
   http://assimp.sourceforge.net/lib_html/data.html

   http://www.tinysg.de/techGuides/tg7_assimpLoader.html


*/

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


void dumpMesh( aiMesh* mesh )
{
    unsigned int numFaces = mesh->mNumFaces;
    unsigned int numVertices = mesh->mNumVertices;
    for(unsigned int i=0 ; i < numVertices ; i++ )
    {
        aiVector3D& v = mesh->mVertices[i] ;
        printf("i %d  xyz %10.3f %10.3f %10.3f \n", i, v.x, v.y, v.z ); 
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



AssimpGeometry::AssimpGeometry(const char* path)
          : 
          m_path(NULL),
          m_importer(new Assimp::Importer()),
          m_aiscene(NULL),
          m_index(0)
{
    if(!path) return ;          
    printf("AssimpGeometry::AssimpGeometry ctor path %s  \n", path);
    m_path = strdup(path);
}

AssimpGeometry::~AssimpGeometry()
{
    // deleting m_importer also deletes the scene (unconfirmed)
    delete m_importer;
    free(m_path);
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



aiNode* AssimpGeometry::getRootNode()
{
   aiNode* root = m_aiscene ? m_aiscene->mRootNode : NULL ;
   return root ; 
}


aiNode* AssimpGeometry::searchNode(const char* query)
{
   aiNode* root = getRootNode();
   if(!root)
   {
       printf("rootnode not defined \n");
       return NULL ; 
   }

   aiNode* node = findNode(query, root, 0); 

   dumpTree(node, 0 );

   return node ; 
}


void AssimpGeometry::dump(aiMaterial* material)
{
    dumpMaterial(material);
}




std::vector<aiNode*>& AssimpGeometry::getSelection()
{
    return m_selection ; 
}

unsigned int AssimpGeometry::select(const char* query)
{
    aiNode* root = m_aiscene ? m_aiscene->mRootNode : NULL ;
    if(!root)
    {
        printf("AssimpGeometry::select root node not defined \n");
        return 0 ; 
    }


    m_index = 0 ; 
    m_selection.clear();

    selectNodes(query, root, 0 );

    printf("AssimpGeometry::select query %s matched %lu nodes \n", query, m_selection.size() ); 

    return m_selection.size();
}


void AssimpGeometry::selectNodes(const char* query, aiNode* node, unsigned int depth )
{

   m_index++ ; 

   const char* name = node->mName.C_Str(); 

   if(strncmp(name,query,strlen(query)) == 0)
   {
      m_selection.push_back(node); 
   }

   for(unsigned int i = 0; i < node->mNumChildren; i++)
   {   
      selectNodes(query, node->mChildren[i], depth + 1 );
   }   
}

void AssimpGeometry::dump(aiMesh* mesh)
{
   dumpMesh(mesh); 
}



