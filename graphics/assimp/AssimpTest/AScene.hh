#ifndef ASCENE_H
#define ASCENE_H

#include <stdlib.h>
#include <assimp/Importer.hpp>    
#include <assimp/scene.h>    
#include <assimp/postprocess.h>        

static unsigned int findNode_index = 0 ; 


void dumpNode(aiNode* node, unsigned int depth, unsigned int sibdex, unsigned int nsibling )
{
   if(!node)
   {
      printf("dumpNode NULL \n");
      return ; 
   }

   unsigned int NumMeshes = node->mNumMeshes ;

   unsigned int NumChildren = node->mNumChildren ;

   const char* name = node->mName.C_Str(); 

   printf("i %5d d %2d m %3d c %3d s %4d/%4d n %s \n", findNode_index, depth, NumMeshes, NumChildren, sibdex, nsibling,  name); 


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


aiNode* findNode(const char* query, aiNode* node, unsigned int depth, unsigned int sibdex, unsigned int nsibling )
{
   if(depth == 0) findNode_index = 0 ; 

   //dumpNode(node, depth, sibdex, nsibling );

   findNode_index++ ;  

   const char* name = node->mName.C_Str(); 

   if(strncmp(name,query,strlen(query)) == 0) return node;


   unsigned int nchild = node->mNumChildren ; 

   for(unsigned int i = 0; i < nchild; i++)
   {   
      aiNode* n = findNode(query, node->mChildren[i], depth + 1, i, nchild );
      if(n) return n ;
   }   
   return NULL ; 
}



void dumpTree(aiNode* node, unsigned int depth, unsigned int sibdex, unsigned int nsibling, bool out )
{

   if(!node)
   {
      printf("dumpTree NULL \n");
      return ; 
   }

   if(depth == 0) findNode_index = 0 ; 

   if(out) dumpNode(node, depth, sibdex, nsibling );
    
   findNode_index++ ;  

   unsigned int nchild =  node->mNumChildren ;

   unsigned int limit = 10 ; 
 
   for(unsigned int i = 0; i < nchild ; i++)
   {   

       bool c_out = ( nchild < limit ) ?  out : out && ( i < limit/2 || i > nchild - limit/2 ) ;

       dumpTree(node->mChildren[i], depth + 1, i , nchild, c_out );
   }   
}




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


      aiNode* searchNode(const char* query)
      {
           aiNode* root = m_scene ? m_scene->mRootNode : NULL ;
           if(!root)
           {
               printf("rootnode not defined \n");
               return NULL ;
           }

           aiNode* node = findNode(query, root, 0, 0, 0);

           printf("dumpTree of found node, query %s \n", query );

           dumpTree(node, 0, 0, 0, true );

           return node ; 
      } 



   private:
       const aiScene* m_scene ; 
       Assimp::Importer m_importer;

};


#endif



