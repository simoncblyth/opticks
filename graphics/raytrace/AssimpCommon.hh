#ifndef ASSIMPCOMMON_H
#define ASSIMPCOMMON_H

struct aiNode ; 
struct aiMesh ;
struct aiMaterial ;

#include <assimp/types.h>


aiNode* findNode(const char* query, aiNode* node, unsigned int depth );
void dumpNode(aiNode* node, unsigned int depth);
void dumpMesh( aiMesh* mesh );
void dumpMaterial( aiMaterial* material );
void dumpTransform(aiMatrix4x4 t);

#endif
