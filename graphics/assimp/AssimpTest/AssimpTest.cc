
#include <stdio.h>
#include <stdlib.h>

#include <assimp/cimport.h>        // Plain-C interface
#include <assimp/scene.h>          // Output data structure
#include <assimp/postprocess.h>    // Post processing flags


void DoTheMaterialProcessing(aiMaterial* material)
{
   //printf("material processing %p \n", material);
}


void DoTheMeshProcessing(aiMesh* mesh)
{
     unsigned int numFaces = mesh->mNumFaces;
     unsigned int numVertices = mesh->mNumVertices;
     //printf("mesh f %d v %d \n", numFaces, numVertices ); 
}

void DoTheNodeProcessing(aiNode* node, unsigned int depth )
{
   aiString name = node->mName ; 

   unsigned int NumChildren = node->mNumChildren ;
   unsigned int NumMeshes   = node->mNumMeshes ;

   printf("node %p d %d c %d m %d n %s \n", node, depth, NumChildren, NumMeshes,  name.C_Str() );

   for(unsigned int i = 0; i < node->mNumChildren; i++)
   {
       aiNode* child = node->mChildren[i] ; 
       DoTheNodeProcessing(child, depth + 1);
   }
}



void DoTheSceneProcessing(const aiScene* scene)
{
   printf("scene processing %p \n", scene);
   printf("scene Flags         %d \n", scene->mFlags );
   printf("scene NumAnimations %d \n", scene->mNumAnimations );
   printf("scene NumCameras    %d \n", scene->mNumCameras );
   printf("scene NumLights     %d \n", scene->mNumLights );
   printf("scene NumMaterials  %d \n", scene->mNumMaterials );
   printf("scene NumMeshes     %d \n", scene->mNumMeshes );
   printf("scene NumTextures   %d \n", scene->mNumTextures );


   for(unsigned int i = 0; i < scene->mNumMeshes; i++)
   {
       aiMesh* mesh = scene->mMeshes[i];
       DoTheMeshProcessing(mesh);
   }

   for(unsigned int i = 0; i < scene->mNumMaterials; i++)
   {
       aiMaterial* material = scene->mMaterials[i];
       DoTheMaterialProcessing(material);
   }

   aiNode* root = scene->mRootNode ; 
   DoTheNodeProcessing(root, 0);
}


void DoTheErrorLogging(const char* msg )
{
   printf("import failed  %s \n", msg);
}


bool DoTheImportThing( const char* pFile)
{
    const aiScene* scene = aiImportFile( pFile, 
          aiProcess_CalcTangentSpace       | 
          aiProcess_Triangulate            |
          aiProcess_JoinIdenticalVertices  |
          aiProcess_SortByPType);

    if( !scene)
    {
        DoTheErrorLogging( aiGetErrorString());
        return false;
    }

    DoTheSceneProcessing( scene);

    aiReleaseImport( scene);
    return true;
}







int main(int argc, char** argv)
{
   const char* key = "DAE_NAME_DYB_NOEXTRA";
   //const char* key = "DAE_NAME_DYB";
   const char* path = getenv(key);
   if(!path) return 1 ; 

   printf("importing key %s path  %s \n", key, path);
   bool ok = DoTheImportThing(path);
   if(!ok) return 2 ; 


   return 0 ;
}
