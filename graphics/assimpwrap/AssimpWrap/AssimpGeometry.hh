#ifndef ASSIMPGEOMETRY_H
#define ASSIMPGEOMETRY_H


class AssimpTree ; 
class AssimpNode ; 

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


class AssimpGeometry 
{
public:
    AssimpGeometry(const char* path);

    virtual ~AssimpGeometry();

    AssimpTree* getTree();

    void import();
    void import(unsigned int flags);
    unsigned int getProcessFlags();
    unsigned int getSceneFlags();
    unsigned int defaultProcessFlags();

    int getQueryMerge();
    int getQueryDepth();
    static const char* identityFilename(char* arg);

    void info();

    void dump();
    void dumpMaterials(const char* msg="AssimpGeometry::dumpMaterials");


    void traverse();

    AssimpNode* getRoot();
    aiMesh* createMergedMesh();

public:
    unsigned int getNumMaterials();
    aiMaterial*  getMaterial(unsigned int index);

public:
    aiVector3D* getLow();
    aiVector3D* getHigh();
    aiVector3D* getCenter();
    aiVector3D* getExtent();
    aiVector3D* getUp();

public:

    unsigned int select(const char* query);

    unsigned int getNumSelected();

    AssimpNode* getSelectedNode(unsigned int i);

    bool isFlatSelection();

protected:

    const aiScene* m_aiscene;

    unsigned int m_index ; 

    unsigned int m_process_flags ; 

private:

    char* m_path ; 

    Assimp::Importer* m_importer;

    AssimpTree* m_tree ; 


};



#endif
