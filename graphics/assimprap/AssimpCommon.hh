#ifndef ASSIMPCOMMON_H
#define ASSIMPCOMMON_H

struct aiNode ; 
struct aiMesh ;
struct aiMaterial ;

#include <assimp/types.h>


aiNode* findNode(const char* query, aiNode* node, unsigned int depth );
void dumpNode(const char* msg, aiNode* node, unsigned int depth, unsigned int index);
bool selectNode(aiNode* node, unsigned int depth, unsigned int index);
void dumpMaterial( aiMaterial* material );
void dumpTransform(const char* msg, aiMatrix4x4 t);

void dumpMesh( aiMesh* mesh );
void copyMesh(aiMesh* dst, aiMesh* src, const aiMatrix4x4& mat );
void meshBounds( aiMesh* mesh );
void meshBounds( aiMesh* mesh, aiVector3D& low, aiVector3D& high );

void dumpProcessFlags(const char* msg, unsigned int flags);
void dumpSceneFlags(const char* msg, unsigned int flags);


#endif
