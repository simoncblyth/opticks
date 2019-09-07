/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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


