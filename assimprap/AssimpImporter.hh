#pragma once

class OpticksQuery ; 

class AssimpTree ; 
class AssimpNode ; 
class AssimpSelection ; 

struct aiScene;
struct aiMesh;
struct aiNode;
struct aiMaterial; 

namespace Assimp
{
    class Importer;
}

#include <assimp/types.h>
#include <vector>

#include "ASIRAP_API_EXPORT.hh"

class ASIRAP_API AssimpImporter 
{
public:
    AssimpImporter(const char* path, int verbosity);
    virtual ~AssimpImporter();
private:
    void init(const char* path);
public:
    AssimpTree* getTree();
    void import();
    void import(unsigned int flags);
    unsigned int getProcessFlags();
    unsigned int getSceneFlags();
    unsigned int defaultProcessFlags();
    //static const char* identityFilename(const char* path, const char* query);
    void Summary(const char* msg="AssimpImporter::Summary");
    void dump();
    void dumpMaterials(const char* msg="AssimpImporter::dumpMaterials");
    void traverse();
    AssimpNode* getRoot();
public:
    unsigned int getNumMaterials();
    aiMaterial*  getMaterial(unsigned int index);

public:
    AssimpSelection* select(OpticksQuery* query);
    aiMesh* createMergedMesh(AssimpSelection* selection);

protected:
    char*           m_path ; 
    int            m_verbosity ; 
    const aiScene* m_aiscene;
    unsigned int   m_index ; 
    unsigned int   m_process_flags ; 
private:
    Assimp::Importer* m_importer;
    AssimpTree*       m_tree ; 
};


