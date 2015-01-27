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


class AssimpGeometry 
{
public:
    AssimpGeometry(const char* path, const char* query );

    virtual ~AssimpGeometry();

    void import();

    void info();

    aiNode* searchNode(const char* query);

private:

    void dumpMaterial(aiMaterial* ai_material);


protected:

    const aiScene* m_aiscene;

    char* m_query ; 

private:

    char* m_path ; 

    Assimp::Importer* m_importer;


};


#endif
