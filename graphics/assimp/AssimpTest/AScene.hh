#ifndef ASCENE_H
#define ASCENE_H

#include <stdlib.h>
#include <assimp/Importer.hpp>    
#include <assimp/scene.h>    
#include <assimp/postprocess.h>        

class AScene 
{
   public:
       AScene(const char* path)
       {   
           m_scene = m_importer.ReadFile( path, 
                     aiProcess_CalcTangentSpace       |   
                     aiProcess_Triangulate            |   
                     aiProcess_JoinIdenticalVertices  |
                     aiProcess_SortByPType);

           if(!m_scene)
           {   
               printf("import error : %s \n", m_importer.GetErrorString() );  
           }   
       }   
       virtual ~AScene()
       {
       }


       void Dump()
       {
          printf("scene %p \n", m_scene);
          printf("scene Flags         %d \n", m_scene->mFlags );
          printf("scene NumAnimations %d \n", m_scene->mNumAnimations );
          printf("scene NumCameras    %d \n", m_scene->mNumCameras );
          printf("scene NumLights     %d \n", m_scene->mNumLights );
          printf("scene NumMaterials  %d \n", m_scene->mNumMaterials );
          printf("scene NumMeshes     %d \n", m_scene->mNumMeshes );
          printf("scene NumTextures   %d \n", m_scene->mNumTextures );
      }




   private:
       const aiScene* m_scene ; 
       Assimp::Importer m_importer;

};


#endif



