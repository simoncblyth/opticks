#include "AssimpCommon.hh"

#include <assimp/scene.h>
#include <assimp/material.h>

#include <sstream>



void split( std::vector<std::string>& elem, const char* line, char delim )
{
    if(line == NULL) return ; 
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim)) elem.push_back(s);
}


aiNode* findNode(const char* query, aiNode* node, unsigned int depth ){
   const char* name = node->mName.C_Str(); 
   if(strncmp(name,query,strlen(query)) == 0) return node;
   for(unsigned int i = 0; i < node->mNumChildren; i++){   
      aiNode* n = findNode(query, node->mChildren[i], depth + 1 );
      if(n) return n ; 
   }   
   return NULL ; 
}

bool selectNode( aiNode* node, unsigned int depth, unsigned int index )
{
   if(!node) return false ;
   unsigned int NumMeshes = node->mNumMeshes ;
   unsigned int NumChildren = node->mNumChildren ;

   //if(NumMeshes > 1)                       //      0 with m>1 : all nodes have 0 or 1 meshes
   //if(NumMeshes == 0 && NumChildren == 1)   //  12231 pv m:0 c:1 
   //if(NumChildren == 0 && NumMeshes == 1)   //   9996 lv m:1 c:0  leaves 
   //if(NumMeshes == 1)                       //  12230 lv m:1 
   //if(NumChildren == 0)                     //   9996 lv c:0    leaves  (all of which have 1 mesh)   
   //if(node->mTransformation.IsIdentity() && NumChildren == 0 && NumMeshes == 1) // 9996 lv c:0 m:1    leaves has identity transform

   if(index < 10) return true ; 
   return false ; 
}


void dumpNode(const char* msg, aiNode* node, unsigned int depth, unsigned int index){
   assert(node);
   unsigned int NumMeshes = node->mNumMeshes ;
   unsigned int NumChildren = node->mNumChildren ;
   const char* name = node->mName.C_Str(); 
   printf("%s i %5d d %2d m %3d c %3d n %s \n", msg, index, depth, NumMeshes, NumChildren, name); 
   //dumpTransform(node->mTransformation);
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



void dumpMesh( aiMesh* mesh ){

    printf("dumpMesh mat %d f %d v %d pt %d hp %d hf %d hn %d htb %d hb %d nuv %d ncc %d \n", 
         mesh->mMaterialIndex, 
         mesh->mNumFaces, 
         mesh->mNumVertices, 
         mesh->mPrimitiveTypes,
         mesh->HasPositions(),
         mesh->HasFaces(),
         mesh->HasNormals(),
         mesh->HasTangentsAndBitangents(),
         mesh->HasBones(),
         mesh->GetNumUVChannels(),
         mesh->GetNumColorChannels()
     );

    for(unsigned int i=0 ; i < mesh->mNumVertices ; i++ )
    {
        aiVector3D& v = mesh->mVertices[i] ;
        if( i == 0 || i == mesh->mNumVertices - 1) 
        printf("i %4d  xyz %10.3f %10.3f %10.3f \n", i, v.x, v.y, v.z ); 
    }

    meshBounds(mesh);
}





void copyMesh(aiMesh* dst, aiMesh* src, const aiMatrix4x4& mat )
{
   // TODO: COMPLETE THE COPY 
   //
   //  /usr/local/env/graphics/assimp/assimp-3.1.1/code/PretransformVertices.cpp
   //  /usr/local/env/graphics/assimp/assimp-3.1.1/code/ColladaLoader.cpp

   dst->mMaterialIndex = src->mMaterialIndex ;
   dst->mPrimitiveTypes = src->mPrimitiveTypes ;

   if (src->HasPositions()) 
   {
       dst->mNumVertices    = src->mNumVertices ;
       dst->mVertices = new aiVector3D[src->mNumVertices];
       for (unsigned int i = 0; i < src->mNumVertices; ++i) {
            dst->mVertices[i] = mat * src->mVertices[i];
       }   
   }   
  
   if (src->HasFaces()) 
   {
       dst->mNumFaces = src->mNumFaces ;
       dst->mFaces = new aiFace[src->mNumFaces];

       for (unsigned int i = 0; i < src->mNumFaces; ++i) {
           dst->mFaces[i] = src->mFaces[i] ;
       }
   }


   if (src->HasNormals() || src->HasTangentsAndBitangents()) 
   {
       aiMatrix4x4 mWorldIT = mat;
       mWorldIT.Inverse().Transpose();

       aiMatrix3x3 m = aiMatrix3x3(mWorldIT);

       if (src->HasNormals()) 
       {
		   dst->mNormals = new aiVector3D[src->mNumVertices];
           for (unsigned int i = 0; i < src->mNumVertices; ++i) {
               dst->mNormals[i] = (m * src->mNormals[i]).Normalize();
           }
       }

       if (src->HasTangentsAndBitangents()) 
       {
		   dst->mTangents = new aiVector3D[src->mNumVertices];
		   dst->mBitangents = new aiVector3D[src->mNumVertices];
           for (unsigned int i = 0; i < src->mNumVertices; ++i) {
               dst->mTangents[i]   = (m * src->mTangents[i]).Normalize();
               dst->mBitangents[i] = (m * src->mBitangents[i]).Normalize();
           }
       }
   }
}


void meshBounds( aiMesh* mesh )
{
    aiVector3D  low( 1e10f, 1e10f, 1e10f);
    aiVector3D high( -1e10f, -1e10f, -1e10f);
    meshBounds(mesh, low, high );
    printf("meshBounds low %10.3f %10.3f %10.3f   high %10.3f %10.3f %10.3f \n", low.x, low.y, low.z, high.x, high.y, high.z );
}


void meshBounds( aiMesh* mesh, aiVector3D& low, aiVector3D& high )
{
    for( unsigned int i = 0; i < mesh->mNumVertices ;++i )
    {    
        aiVector3D v = mesh->mVertices[i];

        low.x = std::min( low.x, v.x);
        low.y = std::min( low.y, v.y);
        low.z = std::min( low.z, v.z);

        high.x = std::max( high.x, v.x);
        high.y = std::max( high.y, v.y);
        high.z = std::max( high.z, v.z);
        //printf("meshBounds low %10.3f %10.3f %10.3f   high %10.3f %10.3f %10.3f \n", low.x, low.y, low.z, high.x, high.y, high.z );
    }
}


