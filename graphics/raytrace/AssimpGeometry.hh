#ifndef ASSIMPGEOMETRY_H
#define ASSIMPGEOMETRY_H

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

    aiNode* getRootNode();
  
    aiNode* searchNode(const char* query);

    std::vector<aiNode*>& getSelection();

    unsigned int select(const char* query);

    void selectNodes(const char* query, aiNode* node, unsigned int depth );

public:

    void dump(aiMaterial* material);

    void dump(aiMesh* mesh);


protected:

    const aiScene* m_aiscene;

    std::vector<aiNode*> m_selection ; 

    unsigned int m_index ; 

private:

    char* m_path ; 

    Assimp::Importer* m_importer;



};


#endif
