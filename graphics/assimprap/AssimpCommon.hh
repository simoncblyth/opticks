#pragma once

struct aiNode ; 
struct aiMesh ;
struct aiMaterial ;

#include <assimp/types.h>
#include "ASIRAP_API_EXPORT.hh"

aiNode* findNode(const char* query, aiNode* node, unsigned int depth );
void ASIRAP_API dumpNode(const char* msg, aiNode* node, unsigned int depth, unsigned int index);
bool ASIRAP_API selectNode(aiNode* node, unsigned int depth, unsigned int index);
void ASIRAP_API dumpMaterial( aiMaterial* material );
void ASIRAP_API dumpTransform(const char* msg, aiMatrix4x4 t);

void ASIRAP_API dumpMesh( aiMesh* mesh );
void ASIRAP_API copyMesh(aiMesh* dst, aiMesh* src, const aiMatrix4x4& mat );
void ASIRAP_API meshBounds( aiMesh* mesh );
void ASIRAP_API meshBounds( aiMesh* mesh, aiVector3D& low, aiVector3D& high );

void ASIRAP_API dumpProcessFlags(const char* msg, unsigned int flags);
void ASIRAP_API dumpSceneFlags(const char* msg, unsigned int flags);


