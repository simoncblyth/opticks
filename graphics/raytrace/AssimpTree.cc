#include "AssimpTree.hh"
#include "AssimpNode.hh"
#include <stdio.h>
#include <assimp/scene.h>

AssimpTree::AssimpTree(const aiScene* scene) 
  : 
  m_scene(scene),
  m_root(NULL),
  m_index(0),
  m_low(NULL),
  m_high(NULL),
  m_center(NULL),
  m_extent(NULL),
  m_query(NULL)
{
   aiNode* top = scene->mRootNode ; 
   m_root = new AssimpNode(top, this);
   m_root->setIndex(0);

   aiMatrix4x4 identity ; 
   wrap( m_root, 0, identity);
   printf("AssimpTree::AssimpTree created tree of %d AssimpNode \n", m_index );
}

AssimpTree::~AssimpTree()
{
}


aiMesh* AssimpTree::getRawMesh(unsigned int meshIndex )
{
    aiMesh* mesh = m_scene->mMeshes[meshIndex];
    return mesh ; 
}

AssimpNode* AssimpTree::getRoot()
{
   return m_root ; 
}

void AssimpTree::traverse()
{
   m_root->traverse();
}


void AssimpTree::wrap(AssimpNode* node, unsigned int depth, aiMatrix4x4 accTransform)
{
   // wrapping the tree
   //
   // accTransform     : accumulated transforms from parentage
   // raw->mTransform  : this node relative to parent 
   // transform        : this node global transform 
   //

   aiNode* raw = node->getRawNode(); 
   aiMatrix4x4 transform = raw->mTransformation * accTransform ;

   node->setIndex(m_index);
   node->setDepth(depth);
   node->copyMeshes(transform);

   m_index++ ; 


   for(unsigned int i = 0; i < raw->mNumChildren; i++) 
   {
       AssimpNode* child = new AssimpNode(raw->mChildren[i], this);

       child->setParent(node);
       node->addChild(child);

       wrap(child, depth + 1, transform);
   }

   // includes bounds of immediate children, to handle 0 mesh levels
   node->updateBounds();

}




void AssimpTree::dumpSelection()
{
   unsigned int n = getNumSelected();
   printf("AssimpTree::dumpSelection n %d \n", n); 
   for(unsigned int i = 0 ; i < n ; i++)
   {
        AssimpNode* node = getSelectedNode(i);
        //node->bounds("AssimpTree::dumpSelection");
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

    selectNodes(query, m_root, 0);

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




void AssimpTree::selectNodes(const char* query, AssimpNode* node, unsigned int depth)
{
   m_index++ ; 
   const char* name = node->getName(); 
   unsigned int index = node->getIndex();

   const char* name_token  = "name:" ;
   const char* index_token = "index:" ;
   const char* range_token = "range:" ;


   if(strncmp(query,name_token, strlen(name_token)) == 0)
   {
       char* q_name = strdup(query+strlen(name_token));
       if(strncmp(name,q_name,strlen(q_name)) == 0)
       {
           m_selection.push_back(node); 
       }
   }  
   else if(strncmp(query,index_token, strlen(index_token)) == 0)
   {
       int q_index = atoi(query+strlen(index_token));
       if( index == q_index )
       {
           m_selection.push_back(node); 
       }
   }
   else if(strncmp(query,range_token, strlen(range_token)) == 0)
   {
       char* q_range = strdup(query+strlen(range_token));
       char* q_delim = strchr(q_range, ':' );
       char* q_right = q_delim + 1  ;
       char* q_left  = q_range ; 
       q_delim = '\0' ; 

       int q_index_left  = atoi(q_left);
       int q_index_right = atoi(q_right);

       free(q_range);

       if( index >= q_index_left && index < q_index_right )
       {
           m_selection.push_back(node); 
       }
     


   } 


   for(unsigned int i = 0; i < node->getNumChildren(); i++) 
   {   
       selectNodes(query, node->getChild(i), depth + 1);
   }   
}






AssimpNode* AssimpTree::searchNode(const char* query)
{
    // non-uniqueness of node names makes this a bit useless anyhow
    return NULL ; 
}




