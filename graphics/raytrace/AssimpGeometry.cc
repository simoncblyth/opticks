#include "AssimpGeometry.hh"
#include "AssimpCommon.hh"
#include "AssimpTree.hh"
#include "AssimpNode.hh"

/*
   http://assimp.sourceforge.net/lib_html/data.html

   http://www.tinysg.de/techGuides/tg7_assimpLoader.html


*/

#include <string.h>
#include <stdlib.h>
#include <sstream>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/material.h>

AssimpGeometry::AssimpGeometry(const char* path)
          : 
          m_path(NULL),
          m_importer(new Assimp::Importer()),
          m_aiscene(NULL),
          m_tree(NULL),
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
        printf("AssimpGeometry::import ERROR : %s \n", m_importer->GetErrorString() );  
    }   

    info();

    m_tree = new AssimpTree(m_aiscene);
}



void AssimpGeometry::traverse()
{
    m_tree->traverse();
}

unsigned int AssimpGeometry::select(const char* query)
{
    return m_tree->select(query);
}

AssimpNode* AssimpGeometry::getRoot()
{
    return m_tree->getRoot();
}

unsigned int AssimpGeometry::getNumSelected()
{
    return m_tree->getNumSelected();
}

AssimpNode* AssimpGeometry::getSelectedNode(unsigned int i)
{
    return m_tree->getSelectedNode(i);
}






