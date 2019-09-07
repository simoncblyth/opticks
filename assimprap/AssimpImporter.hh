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

class OpticksQuery ; 

class AssimpTree ; 
class AssimpNode ; 
class AssimpSelection ; 

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

#include "ASIRAP_API_EXPORT.hh"

class ASIRAP_API AssimpImporter 
{
public:
    AssimpImporter(const char* path, int verbosity);
    virtual ~AssimpImporter();
private:
    void init(const char* path);
public:
    AssimpTree* getTree();
    void import();
    void import(unsigned int flags);
    unsigned int getProcessFlags();
    unsigned int getSceneFlags();
    unsigned int defaultProcessFlags();
    //static const char* identityFilename(const char* path, const char* query);
    void Summary(const char* msg="AssimpImporter::Summary");
    void dump();
    void dumpMaterials(const char* msg="AssimpImporter::dumpMaterials");
    void traverse();
    AssimpNode* getRoot();
public:
    unsigned int getNumMaterials();
    aiMaterial*  getMaterial(unsigned int index);

public:
    AssimpSelection* select(OpticksQuery* query);
    aiMesh* createMergedMesh(AssimpSelection* selection);

protected:
    char*           m_path ; 
    int            m_verbosity ; 
    const aiScene* m_aiscene;
    unsigned int   m_index ; 
    unsigned int   m_process_flags ; 
private:
    Assimp::Importer* m_importer;
    AssimpTree*       m_tree ; 
};


