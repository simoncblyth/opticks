#ifndef ASSIMPCOMMON_H
#define ASSIMPCOMMON_H

struct aiNode ; 
struct aiMesh ;
struct aiMaterial ;

#include <assimp/types.h>


aiNode* findNode(const char* query, aiNode* node, unsigned int depth );
void dumpNode(aiNode* node, unsigned int depth);
void dumpMaterial( aiMaterial* material );
void dumpTransform(aiMatrix4x4 t);

void dumpMesh( aiMesh* mesh );
void copyMesh(aiMesh* dst, aiMesh* src, const aiMatrix4x4& mat );
void meshBounds( aiMesh* mesh );
void meshBounds( aiMesh* mesh, aiVector3D& low, aiVector3D& high );



#endif
