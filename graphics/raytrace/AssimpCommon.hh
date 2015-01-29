#ifndef ASSIMPCOMMON_H
#define ASSIMPCOMMON_H

struct aiNode ; 
struct aiMesh ;
struct aiMaterial ;

#include <assimp/types.h>
#include <vector>
#include <string>

void split( std::vector<std::string>& elem, const char* line, char delim );

aiNode* findNode(const char* query, aiNode* node, unsigned int depth );
void dumpNode(const char* msg, aiNode* node, unsigned int depth, unsigned int index);
bool selectNode(aiNode* node, unsigned int depth, unsigned int index);
void dumpMaterial( aiMaterial* material );
void dumpTransform(const char* msg, aiMatrix4x4 t);

void dumpMesh( aiMesh* mesh );
void copyMesh(aiMesh* dst, aiMesh* src, const aiMatrix4x4& mat );
void meshBounds( aiMesh* mesh );
void meshBounds( aiMesh* mesh, aiVector3D& low, aiVector3D& high );



#endif
