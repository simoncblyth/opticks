
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <cassert>

// assimp-
#include <assimp/Importer.hpp>
#include <assimp/DefaultLogger.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/material.h>

#include "BStr.hh"

// okc--
#include "OpticksQuery.hh"

// assimprap-
#include "AssimpImporter.hh"
#include "AssimpCommon.hh"
#include "AssimpTree.hh"
#include "AssimpNode.hh"
#include "AssimpSelection.hh"

/*
http://assimp.sourceforge.net/lib_html/data.html
http://www.tinysg.de/techGuides/tg7_assimpLoader.html
*/


#include "PLOG.hh"

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
                ::printf("myStream %s", message);
        }
};






AssimpImporter::AssimpImporter(const char* path, int verbosity)
          : 
          m_path(NULL),
          m_verbosity(verbosity),
          m_aiscene(NULL),
          m_index(0),
          m_process_flags(0),
          m_importer(NULL),
          m_tree(NULL)
{
   init(path);
}


void AssimpImporter::init(const char* path)
{
    if(!path) return ;          

    m_importer = new Assimp::Importer() ;

    //printf("AssimpImporter::AssimpImporter ctor path %s  \n", path);
    m_path = strdup(path);

    m_importer->SetPropertyInteger(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION,1);

    DefaultLogger::create("",Logger::VERBOSE);

    //const unsigned int severity = Logger::Info | Logger::Err | Logger::Warn | Logger::Debugging;
    unsigned int severity = Logger::Err | Logger::Warn ;
 
    switch(m_verbosity)
    {
        case 0: severity = Logger::Err | Logger::Warn                                    ; break ; 
        case 1: severity = Logger::Err | Logger::Warn                                    ; break ; 
        case 2: severity = Logger::Err | Logger::Warn | Logger::Info                     ; break ; 
        case 3: severity = Logger::Err | Logger::Warn | Logger::Info | Logger::Debugging ; break ; 
    }

    std::cerr << "AssimpImporter::init"
              << " verbosity " << m_verbosity
              << " severity.Err "  << ( severity & Logger::Err ? "Err" : "no-Err" )
              << " severity.Warn " << ( severity & Logger::Warn ? "Warn" : "no-Warn" )
              << " severity.Info " << ( severity & Logger::Info ? "Info" : "no-Info" )
              << " severity.Debugging " << ( severity & Logger::Debugging ? "Debugging" : "no-Debugging" )
              << std::endl 
              ;

    Assimp::DefaultLogger::get()->attachStream( new myStream(), severity );

    DefaultLogger::get()->debug("debug");
    DefaultLogger::get()->info("info");
    DefaultLogger::get()->warn("warn");
    DefaultLogger::get()->error("error");
}


AssimpImporter::~AssimpImporter()
{
    // deleting m_importer also deletes the scene (unconfirmed)
    delete m_importer;
    free(m_path);
}

AssimpTree* AssimpImporter::getTree()
{
    return m_tree ;
}


void AssimpImporter::Summary(const char* msg)
{

    if(!m_aiscene) return ; 
    
    LOG(info) << msg ;   
    LOG(info) << "AssimpImporter::info m_aiscene " 
               << " NumMaterials " << m_aiscene->mNumMaterials
               << " NumMeshes " << m_aiscene->mNumMeshes ;
  
   /* 
    printf("AssimpImporter::info aiscene %p \n", m_aiscene);
    printf("aiscene Flags         %d \n", m_aiscene->mFlags );
    printf("aiscene NumAnimations %d \n", m_aiscene->mNumAnimations );
    printf("aiscene NumCameras    %d \n", m_aiscene->mNumCameras );
    printf("aiscene NumLights     %d \n", m_aiscene->mNumLights );
    printf("aiscene NumMaterials  %d \n", m_aiscene->mNumMaterials );
    printf("aiscene NumMeshes     %d \n", m_aiscene->mNumMeshes );
    printf("aiscene NumTextures   %d \n", m_aiscene->mNumTextures );
   */
}



/*
const char* AssimpImporter::identityFilename( const char* path, const char* query)
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
    std::string digest = md5digest( query, strlen(query));
    static std::string kfn = insertField( path, '.', -1 , digest.c_str());
    //printf("AssimpImporter::identityFilename\n path %s\n query %s\n digest %s\n kfn %s \n", path, query, digest.c_str(), kfn.c_str() );
    return kfn.c_str();
}
*/


unsigned int AssimpImporter::defaultProcessFlags()
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

void AssimpImporter::import()
{
    import(defaultProcessFlags());
}




unsigned int AssimpImporter::getProcessFlags()
{
   return m_process_flags ; 
}

unsigned int AssimpImporter::getSceneFlags()
{
    return m_aiscene->mFlags ; 
}


void AssimpImporter::import(unsigned int flags)
{
    LOG(info) << "AssimpImporter::import path " << m_path << " flags " << flags ;
    m_process_flags = flags ; 

    assert(m_path);
    m_aiscene = m_importer->ReadFile( m_path, flags );

    if(!m_aiscene)
    {   
        printf("AssimpImporter::import ERROR : \"%s\" \n", m_importer->GetErrorString() );  
        return ;
    }   

    //dumpProcessFlags("AssimpImporter::import", flags);
    //dumpSceneFlags("AssimpImporter::import", m_aiscene->mFlags);

    Summary("AssimpImporter::import DONE");

    m_tree = new AssimpTree(m_aiscene);
}


unsigned int AssimpImporter::getNumMaterials()
{
    return m_aiscene->mNumMaterials ; 
}

aiMaterial* AssimpImporter::getMaterial(unsigned int index)
{
    return m_aiscene->mMaterials[index] ; 
}

void AssimpImporter::traverse()
{
    m_tree->traverse();
}

AssimpSelection* AssimpImporter::select(OpticksQuery* query)
{

    if(!m_tree) 
    {
        printf("AssimpImporter::select no tree \n");
        return 0 ;
    }

    AssimpSelection* selection = new AssimpSelection(m_tree->getRoot(), query);
    //selection->dump();

    if(selection->getNumSelected() == 0)
    {
        printf("AssimpImporter::select WARNING query \"%s\" failed to find any nodes : fallback to adding root  \n", query->getQueryString() );
    }
    return selection ; 
}

AssimpNode* AssimpImporter::getRoot()
{
    return m_tree->getRoot();
}

aiMesh* AssimpImporter::createMergedMesh(AssimpSelection* selection)
{
    return m_tree->createMergedMesh(selection);
}

void AssimpImporter::dump()
{
    m_tree->dump();
}

void AssimpImporter::dumpMaterials(const char* query)
{
    m_tree->dumpMaterials(query);
}



