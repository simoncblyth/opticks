#pragma once

#include <assimp/types.h>
#include <vector>

class AssimpNode ; 
class AssimpRegistry ; 
class AssimpSelection ;

struct aiScene ; 
struct aiMesh ; 
struct aiNode ; 
struct aiMaterial ; 

// strings together a tree of AssimpNode

#include "ASIRAP_API_EXPORT.hh"

class ASIRAP_API AssimpTree {
public:
    AssimpTree(const aiScene* scene);
    virtual ~AssimpTree();
public:
    AssimpNode* getRoot();
    void setRoot(AssimpNode* root);
    const aiScene* getScene();
    aiMesh* getRawMesh(unsigned int meshIndex );
    aiMesh* createMergedMesh(AssimpSelection* selection);  
    void traverse();
    void dump();
public:
    // debug traverse of raw tree
    void traverseRaw(const char* msg="AssimpTree::traverseRaw");
    void traverseRaw(aiNode* raw, std::vector<aiNode*> ancestors);
    void visitRaw(aiNode* raw, std::vector<aiNode*> ancestors);
public:
    void MergeMeshes(AssimpNode* node);
public:
    void dumpMaterials(const char* msg="AssimpTree::dumpMaterials");
    char* getMaterialName(unsigned int materialIndex);    // caller should free
public:
    // wrapping raw tree matching pycollada/g4daenode 
    void traverseWrap(const char* msg="AssimpTree::traverseWrap");
    void traverseWrap(aiNode* node, std::vector<aiNode*> ancestors);
    void visitWrap(std::vector<aiNode*> nodepath);
private:
   const aiScene* m_scene ;  
   AssimpNode* m_root ;  
   AssimpRegistry* m_registry ;  
private:
   unsigned int m_index ; 
   unsigned int m_raw_index ; 
   unsigned int m_wrap_index ; 
   unsigned int m_dump ; 


};


