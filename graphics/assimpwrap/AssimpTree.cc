#include "AssimpTree.hh"
#include "AssimpNode.hh"
#include "AssimpCommon.hh"
#include "AssimpRegistry.hh"
#include "AssimpSelection.hh"

#include <stdio.h>
#include <assimp/scene.h>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


AssimpTree::AssimpTree(const aiScene* scene) 
  : 
  m_scene(scene),
  m_root(NULL),
  m_registry(NULL),
  m_index(0),
  m_raw_index(0),
  m_wrap_index(0),
  m_dump(0)
{
   m_registry = new AssimpRegistry ; 

   traverseWrap();

   //m_registry->summary();
}

AssimpTree::~AssimpTree()
{
}

const aiScene* AssimpTree::getScene()
{
    return m_scene ;
}


void AssimpTree::traverseWrap(const char* msg)
{
   LOG(info) << "AssimpTree::traverseWrap " << msg ; 

   m_wrap_index = 0 ;
   m_dump = 0 ;

   std::vector<aiNode*> ancestors ;

   traverseWrap(m_scene->mRootNode, ancestors);

   LOG(info) << "AssimpTree::traverseWrap m_wrap_index " << m_wrap_index << " m_dump " << m_dump ;  
}

void AssimpTree::traverseWrap(aiNode* node, std::vector<aiNode*> ancestors)
{
   //
   // every node of the tree needs its own nodepath vector
   // this is used to establish a digest for each node, and 
   // a pdigest for the parent 
   //
   // NB the nodepath is complete, ie agnostic regarding visit criteria

   std::vector<aiNode*> nodepath = ancestors ; 
   nodepath.push_back(node) ; 

   if(node->mNumMeshes > 0) visitWrap(nodepath);

   for(unsigned int i = 0; i < node->mNumChildren; i++) traverseWrap(node->mChildren[i], nodepath); 
}


void AssimpTree::visitWrap(std::vector<aiNode*> nodepath)
{
   AssimpNode* wrap = new AssimpNode(nodepath, this) ;       

   wrap->setIndex(m_wrap_index);

   if(m_wrap_index == 0) setRoot(wrap);

   m_registry->add(wrap);

   // hookup relationships via digest matching : works on 1st pass as 
   // parents always exist before children 
   AssimpNode* parent = m_registry->lookup(wrap->getParentDigest());

   if(parent)
   {
       wrap->setParent(parent);
       parent->addChild(wrap);
   }

   aiMatrix4x4 transform = wrap->getGlobalTransform() ; 
   wrap->copyMeshes(transform);

   //if(m_wrap_index == 5000) wrap->ancestors();


   if(0)
   { 
       if(parent) parent->summary("AssimpTree::traW--parent");
       wrap->summary("AssimpTree::traverseWrap");
       dumpTransform("AssimpTree::traverseWrap transform", transform);
   }

   m_wrap_index++;
}


//////////  raw tree traversal attempting to match pycollada/g4daenode.py addressing ///////

void AssimpTree::traverseRaw(const char* msg)
{
   printf("%s\n", msg);

   m_raw_index = 0 ;

   m_dump = 0 ;

   std::vector<aiNode*> ancestors ;

   traverseRaw(m_scene->mRootNode, ancestors );

   printf("%s raw_index %d dump %d \n", msg, m_raw_index, m_dump );
}

void AssimpTree::traverseRaw(aiNode* raw, std::vector<aiNode*> ancestors)
{
   if(raw->mNumMeshes > 0) visitRaw(raw, ancestors );        // all nodes have 0 or 1 meshes for G4DAE Collada 

   ancestors.push_back(raw);

   for(unsigned int i = 0; i < raw->mNumChildren; i++)
   { 
       traverseRaw(raw->mChildren[i], ancestors ); 
   }
}

void AssimpTree::visitRaw(aiNode* raw, std::vector<aiNode*> ancestors )
{
   unsigned int depth = ancestors.size();

   //if(selectNode(raw, depth, m_raw_index))
   {
       for(unsigned int a=0 ; a < ancestors.size() ; a++)
       dumpNode("AssimpTree::v-ancest", ancestors[a], a    , m_raw_index) ;
       dumpNode("AssimpTree::visitRaw", raw         , depth, m_raw_index) ;
       m_dump++ ;
   }
   m_raw_index++;
}




////////////////////////////////////////////////////////////////////////////

aiMesh* AssimpTree::getRawMesh(unsigned int meshIndex )
{
    aiMesh* mesh = m_scene->mMeshes[meshIndex];
    return mesh ; 
}
AssimpNode* AssimpTree::getRoot()
{
   return m_root ; 
}
void AssimpTree::setRoot(AssimpNode* root)
{
   m_root = root ;
}
void AssimpTree::traverse()
{
   m_root->traverse();
}


void AssimpTree::dump()
{
   printf("AssimpTree::dump");
}



aiMesh* AssimpTree::createMergedMesh(AssimpSelection* selection)
{
    // /usr/local/env/graphics/assimp/assimp-3.1.1/code/SceneCombiner.cpp    SceneCombiner::MergeMeshes
    // /usr/local/env/graphics/assimp/assimp-3.1.1/code/OptimizeMeshes.cpp

    aiMesh* out = new aiMesh();

    // 1st past : establish the size
    for(unsigned int i=0 ; i < selection->getNumSelected() ; i++ )
    {
        AssimpNode* node = selection->getSelectedNode(i);
        for(unsigned int i = 0; i < node->getNumMeshes(); i++)
        {   
            aiMesh* mesh = node->getMesh(i);   // these are copied and globally positioned meshes 
            out->mNumVertices += mesh->mNumVertices;
            out->mNumFaces += mesh->mNumFaces;
            out->mNumBones += mesh->mNumBones;
            out->mPrimitiveTypes |= mesh->mPrimitiveTypes;
        } 
    } 

    assert(out->mNumVertices);
    assert(!out->mNumBones);

	aiVector3D* pv ; 
	aiVector3D* pn ; 
	aiVector3D* pt ; 
	aiVector3D* pb ; 
	aiVector3D* px ; 
    aiColor4D* pc ;
    aiFace* pf ;  

    bool first = true ;
    unsigned int n ;
    unsigned int ofs = 0;

    for(unsigned int i=0 ; i < selection->getNumSelected() ; i++ )
    {
        AssimpNode* node = selection->getSelectedNode(i);
        for(unsigned int i = 0; i < node->getNumMeshes(); i++)
        {
            aiMesh* mesh = node->getMesh(i);   // these are copied and globally positioned meshes 

            if(first)
            {
                if(mesh->HasPositions()) 
                {
                    pv = out->mVertices = new aiVector3D[out->mNumVertices];
                }
                if(mesh->HasNormals()) 
                {
                    pn = out->mNormals = new aiVector3D[out->mNumVertices];
                }
                if(mesh->HasTangentsAndBitangents()) 
                {
                    pt = out->mTangents = new aiVector3D[out->mNumVertices];
                    pb = out->mBitangents = new aiVector3D[out->mNumVertices];
                }

                n = 0 ;
		        while (mesh->HasTextureCoords(n))	
                {
			        out->mNumUVComponents[n] = mesh->mNumUVComponents[n];
	 		        px = out->mTextureCoords[n] = new aiVector3D[out->mNumVertices];
                    ++n ;
                } 

                n = 0 ;
		        while (mesh->HasVertexColors(n))	
                {
	 		        pc = out->mColors[n] = new aiColor4D[out->mNumVertices];
                    ++n ;
                } 

                if (out->mNumFaces) 
                {
		            pf = out->mFaces = new aiFace[out->mNumFaces];
                }
                first = false ; 
            }

            if (mesh->HasPositions())
            {
                if(mesh->mVertices)
                {
                    ::memcpy(pv,mesh->mVertices,mesh->mNumVertices*sizeof(aiVector3D));
                }
                pv += mesh->mNumVertices;
            }

            if (mesh->HasNormals())
            {
                if(mesh->mNormals)
                {
                    ::memcpy(pn,mesh->mNormals,mesh->mNumVertices*sizeof(aiVector3D));
                }
                pn += mesh->mNumVertices;
            }
 
            if (mesh->HasTangentsAndBitangents())
            {
                if(mesh->mTangents)
                {
                    ::memcpy(pt,mesh->mTangents,mesh->mNumVertices*sizeof(aiVector3D));
                    ::memcpy(pb,mesh->mBitangents,mesh->mNumVertices*sizeof(aiVector3D));
                }
                pt += mesh->mNumVertices;
                pb += mesh->mNumVertices;
            }

		    n = 0;
		    while ((mesh->HasTextureCoords(n)))
            {
                out->mNumUVComponents[n] = mesh->mNumUVComponents[n];
               
                if(mesh->mTextureCoords[n])
                {
                    ::memcpy(px,mesh->mTextureCoords[n],mesh->mNumVertices*sizeof(aiVector3D));
                }
                px += mesh->mNumVertices;
                ++n;
            }

		    n = 0;
		    while ((mesh->HasVertexColors(n)))
            {
                if(mesh->mColors[n])
                {
                    ::memcpy(pc,mesh->mColors[n],mesh->mNumVertices*sizeof(aiColor4D));
                }
                pc += mesh->mNumVertices;
                ++n;
            }


            if (out->mNumFaces) 
            {
                for (unsigned int m = 0; m < mesh->mNumFaces;++m,++pf)
                {
                    // aiFace assignment operator allocates on heap and does memcpy 
                    aiFace& face = mesh->mFaces[m];   
                    pf->mNumIndices = face.mNumIndices;
                    pf->mIndices = face.mIndices;    //  record the heap pointer into the combi mesh  

                    if (ofs)
                    {
                        for (unsigned int q = 0; q < face.mNumIndices; ++q) face.mIndices[q] += ofs;
                    }
                    face.mIndices = NULL;   // avoid deleting the indices out from underneath 
                }
                ofs += mesh->mNumVertices;
            }


        }   // over meshes (0 or 1 for G4DAE Collada)
    }       // over selected nodes

    return out ;  
}





void AssimpTree::dumpMaterials(const char* query)
{
    printf("%s\n", query );
    for(unsigned int i = 0; i < m_scene->mNumMaterials; i++)
    {   
        aiMaterial* mat = m_scene->mMaterials[i] ;
        aiString name;
        mat->Get(AI_MATKEY_NAME, name);

        printf("AssimpTree::dumpMaterials %s \n", name.C_Str());
        if(strncmp(query, name.C_Str(), strlen(query))==0) dumpMaterial(mat);
    }   
}


char* AssimpTree::getMaterialName(unsigned int materialIndex)
{
    aiMaterial* mat = m_scene->mMaterials[materialIndex] ;
    aiString name;
    mat->Get(AI_MATKEY_NAME, name);
    return strdup(name.C_Str());  
}



