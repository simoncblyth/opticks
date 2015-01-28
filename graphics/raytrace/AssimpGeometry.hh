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

    void import();

    void info();

    void dump();

    void traverse();

    AssimpNode* getRoot();

public:
    aiVector3D* getLow();
    aiVector3D* getHigh();
    aiVector3D* getCenter();
    aiVector3D* getExtent();

public:

    unsigned int select(const char* query);

    unsigned int getNumSelected();

    AssimpNode* getSelectedNode(unsigned int i);

protected:

    const aiScene* m_aiscene;

    unsigned int m_index ; 

private:

    char* m_path ; 

    Assimp::Importer* m_importer;

    AssimpTree* m_tree ; 


};



#endif
