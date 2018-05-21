
#include <iostream>

#include <assimp/types.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>


using namespace Assimp ; 

int main(int argc, char** argv)
{
    if(argc < 2 )
    {
        std::cout << "Expecting argument with path to file to be Assimp imported " << std::endl ; 
        return 0 ; 
    }
 
    const char* path = argv[1] ; 
    std::cout << argv[0] << " importing " << path << std::endl ; 

    Assimp::Importer importer ; 
    unsigned flags = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices ;
    const aiScene* m_aiscene = importer.ReadFile(path, flags);

    std::cout << " scene " << m_aiscene << std::endl ; 
   
    printf("AssimpImporter::info aiscene %p \n", m_aiscene);
    printf("aiscene Flags         %d \n", m_aiscene->mFlags );
    printf("aiscene NumAnimations %d \n", m_aiscene->mNumAnimations );
    printf("aiscene NumCameras    %d \n", m_aiscene->mNumCameras );
    printf("aiscene NumLights     %d \n", m_aiscene->mNumLights );
    printf("aiscene NumMaterials  %d \n", m_aiscene->mNumMaterials );
    printf("aiscene NumMeshes     %d \n", m_aiscene->mNumMeshes );
    printf("aiscene NumTextures   %d \n", m_aiscene->mNumTextures );



    return 0 ; 
}
