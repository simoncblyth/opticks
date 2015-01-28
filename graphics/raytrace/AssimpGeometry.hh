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

#include <vector>

class AssimpGeometry 
{
public:
    AssimpGeometry(const char* path);

    virtual ~AssimpGeometry();

    void import();

    void info();

    void traverse();

    AssimpNode* getRoot();

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
