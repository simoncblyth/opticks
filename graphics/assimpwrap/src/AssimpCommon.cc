#include "AssimpCommon.hh"
#include "md5digest.h"
#include <assimp/scene.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>

#include <sstream>



void split( std::vector<std::string>& elem, const char* line, char delim )
{
    if(line == NULL) return ; 
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim)) elem.push_back(s);
}


std::string join(std::vector<std::string>& elem, char delim )
{
    typedef std::vector<std::string> Vec_t ;
    std::stringstream ss ;
    for(size_t i=0 ; i < elem.size() ; ++i)
    {
        ss << elem[i] ;
        if( i < elem.size() - 1) ss << delim ;
    }
    return ss.str();
}

void removeField(char* dest, const char* line, char delim, int index )
{

    //  
    //   split the line with the delim
    //   then reassemble skipping the field pointed to by rfield
    //  
    //    For example the below line with delim '.' and rfield -2
    //  
    //       /path/to/geometry.dae.noextra.abcdefghijklmnopqrstuvwxyz.dae
    //       /path/to/geometry.dae.noextra.dae
    //  

    //printf("AssimpCommon::removeField line %s delim %c index %d \n", line, delim, index);


    std::vector<std::string> elem ;
    split(elem, line, delim);

    //printf("AssimpCommon:: size %lu \n", elem.size() );

    if(index >= 0 && index < elem.size())
    {
        elem.erase( elem.begin() + index);
    }
    else if( index < 0 && -index < elem.size())
    {
        elem.erase( elem.end() + index );
    }
    else
    {
        printf("removeField line %s delim %c index %d : invalid index \n", line, delim, index );
    }
    std::string j = join(elem, delim);

    //printf("AssimpCommon::removeField j %s \n", j.c_str() );
    strcpy( dest, j.c_str()) ;
}


std::string insertField(const char* line, char delim, int index, const char* field)
{
    std::vector<std::string> elem ;
    split(elem, line, delim);    

    std::string s(field);

    if(index >= 0 && index < elem.size())
    {   
        elem.insert( elem.begin() + index, s); 
    }   
    else if( index < 0 && -index < elem.size())
    {   
        elem.insert( elem.end() + index, s );
    }   
    else
    {   
        printf("insertField line %s delim %c index %d : invalid index \n", line, delim, index );
    }   
    return join(elem, delim); 
}



std::string md5digest( const char* buffer, int len )
{
    char* out = md5digest_str2md5(buffer, len);
    std::string digest(out);
    free(out);
    return digest;
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

void dumpTransform(const char* msg, aiMatrix4x4 t)
{
   printf("%s\n", msg);
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
   //
   //  hmm maybe can use
   //   /usr/local/env/graphics/assimp/assimp-3.1.1/code/SceneCombiner.cpp
   // 

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

void dumpSceneFlags(const char* msg, unsigned int flags)
{
    printf("dumpSceneFlags %s flags 0x%x \n", msg, flags);

    if(flags & AI_SCENE_FLAGS_INCOMPLETE)
    printf("AI_SCENE_FLAGS_INCOMPLETE\n");

    if(flags & AI_SCENE_FLAGS_VALIDATED)
    printf("AI_SCENE_FLAGS_VALIDATED\n");

    if(flags & AI_SCENE_FLAGS_VALIDATION_WARNING)
    printf("AI_SCENE_FLAGS_VALIDATION_WARNING\n");

    if(flags & AI_SCENE_FLAGS_NON_VERBOSE_FORMAT)
    printf("AI_SCENE_FLAGS_NON_VERBOSE_FORMAT\n");

    if(flags & AI_SCENE_FLAGS_TERRAIN)
    printf("AI_SCENE_FLAGS_TERRAIN\n");
}


void dumpProcessFlags(const char* msg, unsigned int flags)
{
    printf("dumpProcessFlags %s flags 0x%x \n", msg, flags);

    if(flags & aiProcess_CalcTangentSpace)      
    printf("aiProcess_CalcTangentSpace\n"); 

    if(flags & aiProcess_JoinIdenticalVertices) 
    printf("aiProcess_JoinIdenticalVertices\n");

    if(flags & aiProcess_MakeLeftHanded ) 
    printf("aiProcess_MakeLeftHanded\n");

    if(flags & aiProcess_Triangulate)
    printf("aiProcess_Triangulate\n");

    if(flags&aiProcess_RemoveComponent)
    printf("aiProcess_RemoveComponent\n");

    if(flags & aiProcess_GenNormals) 
    printf("aiProcess_GenNormals\n");

    if(flags & aiProcess_GenSmoothNormals) 
    printf("aiProcess_GenSmoothNormals\n");

    if(flags & aiProcess_SplitLargeMeshes) 
    printf("aiProcess_SplitLargeMeshes\n");

    if(flags & aiProcess_PreTransformVertices) 
    printf("aiProcess_PreTransformVertices\n");

    if(flags & aiProcess_LimitBoneWeights)
    printf("aiProcess_LimitBoneWeights\n");

    if(flags & aiProcess_ValidateDataStructure) 
    printf("aiProcess_ValidateDataStructure\n");

    if(flags & aiProcess_ImproveCacheLocality) 
    printf("aiProcess_ImproveCacheLocality\n");

    if(flags & aiProcess_RemoveRedundantMaterials) 
    printf("aiProcess_RemoveRedundantMaterials\n");

    if(flags & aiProcess_FixInfacingNormals) 
    printf("aiProcess_FixInfacingNormals\n");

    if(flags & aiProcess_SortByPType) 
    printf("aiProcess_SortByPType\n");

    if(flags & aiProcess_FindDegenerates) 
    printf("aiProcess_FindDegenerates\n");

    if(flags & aiProcess_FindInvalidData)
    printf("aiProcess_FindInvalidData\n");

    if(flags & aiProcess_GenUVCoords)
    printf("aiProcess_GenUVCoords\n");

    if(flags & aiProcess_TransformUVCoords)
    printf("aiProcess_TransformUVCoords\n");

    if(flags & aiProcess_FindInstances) 
    printf("aiProcess_FindInstances\n");

    if(flags & aiProcess_OptimizeMeshes)
    printf("aiProcess_OptimizeMeshes\n");

    if(flags & aiProcess_OptimizeGraph)
    printf("aiProcess_OptimizeGraph\n");

    if(flags & aiProcess_FlipUVs) 
    printf("aiProcess_FlipUVs\n");

    if(flags & aiProcess_FlipWindingOrder) 
    printf("aiProcess_FlipWindingOrder\n");

    if(flags & aiProcess_SplitByBoneCount) 
    printf("aiProcess_SplitByBoneCount\n");

    if(flags & aiProcess_Debone) 
    printf("aiProcess_Debone\n");
}




