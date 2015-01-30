#include "AssimpTree.hh"
#include "AssimpNode.hh"
#include "AssimpCommon.hh"
#include "AssimpRegistry.hh"
#include <stdio.h>
#include <assimp/scene.h>

AssimpTree::AssimpTree(const aiScene* scene) 
  : 
  m_scene(scene),
  m_root(NULL),
  m_registry(NULL),
  m_index(0),
  m_raw_index(0),
  m_wrap_index(0),
  m_dump(0),
  m_low(NULL),
  m_high(NULL),
  m_center(NULL),
  m_extent(NULL),
  m_up(NULL),
  m_query(NULL),
  m_query_name(NULL),
  m_query_index(0), 
  m_is_flat_selection(false)
{

   m_registry = new AssimpRegistry ; 

   traverseWrap();

   m_registry->summary();

   //m_root->traverse();
}

AssimpTree::~AssimpTree()
{
}



void AssimpTree::traverseWrap(const char* msg)
{
   printf("%s\n", msg);
   m_wrap_index = 0 ;
   m_dump = 0 ;

   std::vector<aiNode*> ancestors ;

   traverseWrap(m_scene->mRootNode, ancestors);

   printf("%s wrap_index %d dump %d \n", msg, m_wrap_index, m_dump );
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

   // hookup relationships via digest matching 
   AssimpNode* parent = m_registry->lookup(wrap->getParentDigest());

   if(parent)
   {
       wrap->setParent(parent);
       parent->addChild(wrap);
   }

   aiMatrix4x4 transform = wrap->getGlobalTransform() ; 
   wrap->copyMeshes(transform);

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

   if(selectNode(raw, depth, m_raw_index))
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


void AssimpTree::dumpSelection()
{
   unsigned int n = getNumSelected();
   printf("AssimpTree::dumpSelection n %d \n", n); 

   for(unsigned int i = 0 ; i < n ; i++)
   {
        if( i > 10 ) break ; 
        AssimpNode* node = getSelectedNode(i);
        node->summary("AssimpTree::dumpSelection");
        node->bounds("AssimpTree::dumpSelection");
   } 
}


unsigned int AssimpTree::getNumSelected()
{
    return m_selection.size();
}

AssimpNode* AssimpTree::getSelectedNode(unsigned int i)
{
    return i < m_selection.size() ? m_selection[i] : NULL ; 
}


void AssimpTree::findBounds()
{
    aiVector3D  up( 0.f, 1.f, 0.f);
    aiVector3D  low( 1e10f, 1e10f, 1e10f);
    aiVector3D high( -1e10f, -1e10f, -1e10f);

   unsigned int n = getNumSelected();
   printf("AssimpTree::findBounds n %d \n", n); 
   for(unsigned int i = 0 ; i < n ; i++)
   {
       AssimpNode* node = getSelectedNode(i);
       findBounds( node, low, high );
   } 

   if( n > 0)
   { 
      delete m_low ; 
      m_low  = new aiVector3D(low);

      delete m_high ;
      m_high = new aiVector3D(high);

      delete m_center ;
      m_center = new aiVector3D((high+low)/2.f);

      delete m_extent ;
      m_extent = new aiVector3D(high-low);

      delete m_up ;
      m_up = new aiVector3D(up);

   }
}


void AssimpTree::findBounds(AssimpNode* node, aiVector3D& low, aiVector3D& high )
{
    aiVector3D* nlow  = node->getLow();
    aiVector3D* nhigh = node->getHigh();

    if(nlow && nhigh)
    {
        low.x = std::min( low.x, nlow->x);
        low.y = std::min( low.y, nlow->y);
        low.z = std::min( low.z, nlow->z);

        high.x = std::max( high.x, nhigh->x);
        high.y = std::max( high.y, nhigh->y);
        high.z = std::max( high.z, nhigh->z);
   
   } 
}


void AssimpTree::MergeMeshes(AssimpNode* node)
{
    // /usr/local/env/graphics/assimp/assimp-3.1.1/code/SceneCombiner.cpp    SceneCombiner::MergeMeshes
    // /usr/local/env/graphics/assimp/assimp-3.1.1/code/OptimizeMeshes.cpp
}


unsigned int AssimpTree::select(const char* query)
{
    if(!m_root)
    {
        printf("AssimpTree::select ERROR tree not yet wrapped \n");
        return 0 ;
    } 

    free(m_query) ;

    m_query = strdup(query);

    m_index = 0 ; 

    m_selection.clear();

    parseQuery(query);    

    selectNodes(m_root, 0);

    findBounds();

    return m_selection.size();

}


void AssimpTree::dump()
{
    printf("AssimpTree::dump query %s selection matched %lu nodes \n", m_query, m_selection.size() ); 
    bounds();
    dumpSelection();
}

void AssimpTree::bounds()
{
    if(m_center)  printf("AssimpTree::bounds cen  %10.3f %10.3f %10.3f \n", m_center->x, m_center->y, m_center->z );
    if(m_low)  printf("AssimpTree::bounds low  %10.3f %10.3f %10.3f \n", m_low->x, m_low->y, m_low->z );
    if(m_high) printf("AssimpTree::bounds high %10.3f %10.3f %10.3f \n", m_high->x, m_high->y, m_high->z );
    if(m_extent) 
               printf("AssimpTree::bounds ext  %10.3f %10.3f %10.3f \n", m_extent->x, m_extent->y, m_extent->z );
}


aiVector3D* AssimpTree::getLow()
{
    return m_low ; 
}
aiVector3D* AssimpTree::getHigh()
{
    return m_high ; 
}
aiVector3D* AssimpTree::getCenter()
{
    return m_center ; 
}
aiVector3D* AssimpTree::getExtent()
{
    return m_extent ; 
}
aiVector3D* AssimpTree::getUp()
{
    return m_up ; 
}


void AssimpTree::addToSelection(AssimpNode* node)
{
   m_selection.push_back(node);
}





void AssimpTree::parseQuery(const char* query)
{

   const char* name_token  = "name:" ;
   const char* index_token = "index:" ;
   const char* range_token = "range:" ;

   if(strncmp(query,name_token, strlen(name_token)) == 0)
   {
       m_query_name = strdup(query+strlen(name_token));
   }  
   else if(strncmp(query,index_token, strlen(index_token)) == 0)
   {
       m_query_index = atoi(query+strlen(index_token));
   }
   else if(strncmp(query,range_token, strlen(range_token)) == 0)
   {
       std::vector<std::string> elem ; 
       split(elem, query+strlen(range_token), ':'); 
       assert(elem.size() == 2);
       m_query_range.clear();
       for(int i=0 ; i<elem.size() ; ++i) m_query_range.push_back( atoi(elem[i].c_str()) ) ;

       m_is_flat_selection = true ; 
  } 
}



bool AssimpTree::isFlatSelection()
{
   return m_is_flat_selection ; 
}





void AssimpTree::selectNodes(AssimpNode* node, unsigned int depth)
{
   m_index++ ; 
   const char* name = node->getName(); 
   unsigned int index = node->getIndex();


   if(m_query_name)
   {
       if(strncmp(name,m_query_name,strlen(m_query_name)) == 0)
       {
           m_selection.push_back(node); 
       }
   }
   else if (m_query_index != 0)
   {
       if( index == m_query_index )
       {
           m_selection.push_back(node); 
       }
   }
   else if(m_query_range.size() == 2)
   {
       if( index >= m_query_range[0] && index < m_query_range[1] )
       {
           m_selection.push_back(node); 
       }
   }
   

   for(unsigned int i = 0; i < node->getNumChildren(); i++) selectNodes(node->getChild(i), depth + 1);
}




