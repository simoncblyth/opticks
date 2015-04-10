#include "AssimpGeometry.hh"
#include "AssimpCommon.hh"
#include "AssimpTree.hh"
#include "AssimpNode.hh"
#include "AssimpSelection.hh"

/*
   http://assimp.sourceforge.net/lib_html/data.html

   http://www.tinysg.de/techGuides/tg7_assimpLoader.html


*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>

#include "assert.h"

#include <assimp/Importer.hpp>
#include <assimp/DefaultLogger.hpp>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/material.h>


using namespace Assimp ; 


class myStream : public LogStream
{
public:
        myStream()
        {
        }
        
        ~myStream()
        {
        }
        void write(const char* message)
        {
                ::printf("%s", message);
        }
};


AssimpGeometry::AssimpGeometry(const char* path)
          : 
          m_path(NULL),
          m_importer(new Assimp::Importer()),
          m_aiscene(NULL),
          m_tree(NULL),
          m_process_flags(0),
          m_index(0)
{
    if(!path) return ;          
    printf("AssimpGeometry::AssimpGeometry ctor path %s  \n", path);
    m_path = strdup(path);

    DefaultLogger::create("",Logger::VERBOSE);

    //const unsigned int severity = Logger::Info | Logger::Err | Logger::Warn | Logger::Debugging;
    const unsigned int severity = Logger::Err | Logger::Warn ;
    
    Assimp::DefaultLogger::get()->attachStream( new myStream(), severity );

    DefaultLogger::get()->info("this is my info-call");
}

AssimpGeometry::~AssimpGeometry()
{
    // deleting m_importer also deletes the scene (unconfirmed)
    delete m_importer;
    free(m_path);
}

AssimpTree* AssimpGeometry::getTree()
{
    return m_tree ;
}


void AssimpGeometry::info()
{
    if(!m_aiscene) return ; 
    printf("AssimpGeometry::info NumMaterial %d NumMeshes %d  \n", m_aiscene->mNumMaterials, m_aiscene->mNumMeshes );
  
   /* 
    printf("AssimpGeometry::info aiscene %p \n", m_aiscene);
    printf("aiscene Flags         %d \n", m_aiscene->mFlags );
    printf("aiscene NumAnimations %d \n", m_aiscene->mNumAnimations );
    printf("aiscene NumCameras    %d \n", m_aiscene->mNumCameras );
    printf("aiscene NumLights     %d \n", m_aiscene->mNumLights );
    printf("aiscene NumMaterials  %d \n", m_aiscene->mNumMaterials );
    printf("aiscene NumMeshes     %d \n", m_aiscene->mNumMeshes );
    printf("aiscene NumTextures   %d \n", m_aiscene->mNumTextures );
   */
}


const char* AssimpGeometry::identityFilename( char* arg )
{
    // Used when geometry is loaded using options: -g/--g4dae path  
    //
    // #. command line argument real path is converted into one 
    //    incorporating the digest of geometry selecting envvar 
     //
    //    This kludge makes the accelcache path incorporate the digest 
    //    thus can get caching as vary the envvar that 
    //    selects different geometries, while still ony having a single 
    //    geometry file.
    //
    char* query = getenv("RAYTRACE_QUERY") ;
    std::string digest = md5digest( query, strlen(query));
    static std::string kfn = insertField( arg, '.', -1 , digest.c_str());
    printf("AssimpGeometry::identityFilename\n arg %s\n query %s\n digest %s\n kfn %s \n", arg, query, digest.c_str(), kfn.c_str() );
    return kfn.c_str();
}



unsigned int AssimpGeometry::defaultProcessFlags()
{
    unsigned int flags = 0 ;

    flags |=  aiProcess_CalcTangentSpace ;
    flags |=  aiProcess_Triangulate ;
    flags |=  aiProcess_JoinIdenticalVertices ;
    flags |=  aiProcess_SortByPType ;

    //flags |=  aiProcess_GenNormals ;  // ADD Apr8 2015

    // above flags were used initially 
    // changing flags invalidates the accelcache, due to different geometry : so they are doing something 
    //
    // TODO: flags and maxdepth setting need to feed into the accelcache name 
    //
   //  flags |=  aiProcess_OptimizeMeshes  ;
    // flags |=  aiProcess_OptimizeGraph ;

    return flags ; 
}

void AssimpGeometry::import()
{
    import(defaultProcessFlags());
}




unsigned int AssimpGeometry::getProcessFlags()
{
   return m_process_flags ; 
}

unsigned int AssimpGeometry::getSceneFlags()
{
    return m_aiscene->mFlags ; 
}


void AssimpGeometry::import(unsigned int flags)
{
    printf("AssimpGeometry::import path %s flags 0x%x \n", m_path, flags  );

    m_process_flags = flags ; 

    assert(m_path);
    m_aiscene = m_importer->ReadFile( m_path, flags );

    if(!m_aiscene)
    {   
        printf("AssimpGeometry::import ERROR : \"%s\" \n", m_importer->GetErrorString() );  
        return ;
    }   

    dumpProcessFlags("AssimpGeometry::import", flags);
    dumpSceneFlags("AssimpGeometry::import", m_aiscene->mFlags);

    info();

    m_tree = new AssimpTree(m_aiscene);
}


unsigned int AssimpGeometry::getNumMaterials()
{
    return m_aiscene->mNumMaterials ; 
}

aiMaterial* AssimpGeometry::getMaterial(unsigned int index)
{
    return m_aiscene->mMaterials[index] ; 
}

void AssimpGeometry::traverse()
{
    m_tree->traverse();
}

AssimpSelection* AssimpGeometry::select(const char* query)
{

    if(!m_tree) 
    {
        printf("AssimpGeometry::select no tree \n");
        return 0 ;
    }

    AssimpSelection* selection = new AssimpSelection(m_tree->getRoot(), query);
    selection->dump();

    if(selection->getNumSelected() == 0)
    {
        printf("AssimpGeometry::select WARNING query \"%s\" failed to find any nodes : fallback to adding root  \n", query );
    }
    return selection ; 
}

AssimpNode* AssimpGeometry::getRoot()
{
    return m_tree->getRoot();
}

aiMesh* AssimpGeometry::createMergedMesh(AssimpSelection* selection)
{
    return m_tree->createMergedMesh(selection);
}

void AssimpGeometry::dump()
{
    m_tree->dump();
}

void AssimpGeometry::dumpMaterials(const char* query)
{
    m_tree->dumpMaterials(query);
}



