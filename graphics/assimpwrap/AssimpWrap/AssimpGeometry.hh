#ifndef ASSIMPGEOMETRY_H
#define ASSIMPGEOMETRY_H


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

    static const char* identityFilename(char* arg);

    void info();

    void dump();
    void dumpMaterials(const char* msg="AssimpGeometry::dumpMaterials");


    void traverse();

    AssimpNode* getRoot();

public:
    unsigned int getNumMaterials();
    aiMaterial*  getMaterial(unsigned int index);

public:

    AssimpSelection* select(const char* query);

    aiMesh* createMergedMesh(AssimpSelection* selection);

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
