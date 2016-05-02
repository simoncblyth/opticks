#include "AssimpSelection.hh"
#include "AssimpNode.hh"
#include "AssimpCommon.hh"
#include "AssimpRegistry.hh"

#include <algorithm>
#include <stdio.h>
#include <assimp/scene.h>

// npy-
#include "stringutil.hpp"
#include "NLog.hpp"



const char* AssimpSelection::UNDEFINED_ = "undefined" ; 
const char* AssimpSelection::NAME_ = "name" ; 
const char* AssimpSelection::INDEX_ = "index" ; 
const char* AssimpSelection::MERGE_ = "merge" ; 
const char* AssimpSelection::DEPTH_ = "depth" ; 
const char* AssimpSelection::RANGE_ = "range" ; 



void AssimpSelection::init()
{
    parseQuery(m_query);    
    dumpQuery("AssimpSelection::init dumpQuery");    

    m_selection.clear();

    LOG(info) << "AssimpSelection::AssimpSelection"
              << " before SelectNodes " 
              << " m_query " << m_query 
              << " m_query_name " <<  (m_query_name ? m_query_name : "NULL" )
              << " m_query_index " <<  m_query_index 
              ;

    selectNodes(m_root, 0, false);

    LOG(info) << "AssimpSelection::AssimpSelection"
              << " after SelectNodes " 
              << " m_selection size " << m_selection.size()
              << " out of m_count " << m_count 
              ;

    assert(m_selection.size() > 0 );

    findBounds();
}

AssimpSelection::~AssimpSelection()
{
   free(m_query);
}


bool AssimpSelection::contains(AssimpNode* node)
{
    return std::find(m_selection.begin(), m_selection.end(), node ) != m_selection.end() ;  
}

void AssimpSelection::dumpSelection()
{
   unsigned int n = getNumSelected();
   printf("AssimpSelection::dumpSelection n %d \n", n); 

   /*
   for(unsigned int i = 0 ; i < n ; i++)
   {
        if( i > 10 ) break ; 
        AssimpNode* node = getSelectedNode(i);
        node->summary("AssimpSelection::dumpSelection");
        node->bounds("AssimpSelection::dumpSelection");
   } 
   */
}


unsigned int AssimpSelection::getNumSelected()
{
    return m_selection.size();
}

AssimpNode* AssimpSelection::getSelectedNode(unsigned int i)
{
    return i < m_selection.size() ? m_selection[i] : NULL ; 
}

void AssimpSelection::addToSelection(AssimpNode* node)
{
     m_selection.push_back(node);
}

void AssimpSelection::parseQuery(const char* query)
{
   // "name:helo,index:yo,range:10"
   // split at "," and then extract values beyond the "token:" 

   std::vector<std::string> elem ; 
   split(elem, query, ',');

   for(unsigned int i=0 ; i < elem.size() ; i++ ) parseQueryElement( elem[i].c_str() );

   if(elem.size() == 0)
   {
      m_no_selection = true ; 
   }

   LOG(info) << "AssimpSelection::parseQuery" 
             << " query:[" << query << "]"
             << " elements:" << elem.size()  
             << " queryType:" << getQueryTypeString()
             ;


}

void AssimpSelection::dumpQuery(const char* msg)
{
    LOG(info) << msg 
              << " queryType " << getQueryTypeString() ;

   if(m_query_type == RANGE)
   {
       size_t nrange = m_query_range.size() ;
       std::cout << " nrange " << nrange ;
       for(unsigned int i=0 ; i < nrange ; i++)
       {
          std::cout << " : " << m_query_range[i] ;
       }
       std::cout << std::endl ; 
   } 
           
}

AssimpSelection::AssimpQuery_t AssimpSelection::getQueryType()
{
    return m_query_type ; 
}

const char* AssimpSelection::getQueryTypeString()
{
   const char* type = UNDEFINED_ ;
   switch(m_query_type)
   {
      case UNDEFINED: type=UNDEFINED_ ;break; 
      case NAME     : type=NAME_      ;break; 
      case INDEX    : type=INDEX_     ;break; 
      case RANGE    : type=RANGE_     ;break; 
      case MERGE    : type=MERGE_     ;break; 
      case DEPTH    : type=DEPTH_     ;break; 
      default       : type=UNDEFINED_ ;break; 
   }
   return type ; 
}



void AssimpSelection::parseQueryElement(const char* query)
{
   const char* name_token  = "name:" ;
   const char* index_token = "index:" ;
   const char* range_token = "range:" ;
   const char* merge_token = "merge:" ;
   const char* depth_token = "depth:" ;

   m_query_type = UNDEFINED ; 

   if(strncmp(query,name_token, strlen(name_token)) == 0)
   {
       m_query_type = NAME ; 
       m_query_name = strdup(query+strlen(name_token));
   }  
   else if(strncmp(query,index_token, strlen(index_token)) == 0)
   {
       m_query_type = INDEX ; 
       int query_index = atoi(query+strlen(index_token));
       assert(query_index > -1);
       m_query_index = query_index ;
   }
   else if(strncmp(query,merge_token, strlen(merge_token)) == 0)
   {
       m_query_type = MERGE ; 
       int query_merge = atoi(query+strlen(merge_token));
       assert(query_merge > -1);
       m_query_merge = query_merge ;
   }
   else if(strncmp(query,depth_token, strlen(depth_token)) == 0)
   {
       m_query_type = DEPTH ; 
       int query_depth = atoi(query+strlen(depth_token));
       assert(query_depth > -1);
       m_query_depth = query_depth ;
   }
   else if(strncmp(query,range_token, strlen(range_token)) == 0)
   {
       m_query_type = RANGE ; 
       std::vector<std::string> elem ; 
       split(elem, query+strlen(range_token), ':'); 
       assert(elem.size() == 2);
       //m_query_range.clear();
       for(unsigned int i=0 ; i<elem.size() ; ++i)
       {
           int query_range_elem = atoi(elem[i].c_str());
           assert(query_range_elem > -1 );
           m_query_range.push_back(query_range_elem) ;
       }
       m_flat_selection = true ; 
  } 
}




void AssimpSelection::selectNodes(AssimpNode* node, unsigned int depth, bool rselect )
{
   // recursive traverse, adding nodes fulfiling the selection
   // criteria into m_selection 

   m_count++ ; 
   m_index++ ; 
   const char* name = node->getName(); 
   unsigned int index = node->getIndex();

   if(m_count < 10)
   {
      LOG(debug) << "AssimpSelection::selectNodes "
                << " m_count " << m_count 
                << " m_index " << m_index
                << " index " << index 
                << " name " << name 
                ;
   } 

   if(m_no_selection)
   {
       m_selection.push_back(node); 
   }
   else if(m_query_name)
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
           rselect = true ;   
           // kick-off recursive select, note it then never 
           // gets turned off, but it only causes selection for depth within range 
       }
       else if ( rselect ) 
       {
           if( m_query_depth > 0 && depth < m_query_depth ) m_selection.push_back(node); 
       }
   }
   else if(m_query_range.size() > 0)
   {
       assert(m_query_range.size() % 2 == 0); 
       for(unsigned int i=0 ; i < m_query_range.size()/2 ; i++ )
       {
           if( index >= m_query_range[i*2+0] && index < m_query_range[i*2+1] ) m_selection.push_back(node); 
       }
   }

   for(unsigned int i = 0; i < node->getNumChildren(); i++) selectNodes(node->getChild(i), depth + 1, rselect );
}



void AssimpSelection::findBounds()
{
    aiVector3D  up( 0.f, 1.f, 0.f);
    aiVector3D  low( 1e10f, 1e10f, 1e10f);
    aiVector3D high( -1e10f, -1e10f, -1e10f);

   unsigned int n = getNumSelected();

   LOG(info) << "AssimpSelection::findBounds " 
             << " NumSelected " << n ;

   for(unsigned int i = 0 ; i < n ; i++)
   {
       AssimpNode* node = getSelectedNode(i);
       findBounds( node, low, high );
   } 

   //assert(n>0);

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


void AssimpSelection::findBounds(AssimpNode* node, aiVector3D& low, aiVector3D& high )
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



void AssimpSelection::dump()
{
    printf("AssimpSelection::dump query %s selection matched %lu nodes \n", m_query, m_selection.size() ); 
    bounds();
    //dumpSelection();
}

void AssimpSelection::bounds()
{
    if(m_center)  printf("AssimpSelection::bounds cen  %10.3f %10.3f %10.3f \n", m_center->x, m_center->y, m_center->z );
    if(m_low)  printf("AssimpSelection::bounds low  %10.3f %10.3f %10.3f \n", m_low->x, m_low->y, m_low->z );
    if(m_high) printf("AssimpSelection::bounds high %10.3f %10.3f %10.3f \n", m_high->x, m_high->y, m_high->z );
    if(m_extent) 
               printf("AssimpSelection::bounds ext  %10.3f %10.3f %10.3f \n", m_extent->x, m_extent->y, m_extent->z );
}


aiVector3D* AssimpSelection::getLow()
{
    return m_low ; 
}
aiVector3D* AssimpSelection::getHigh()
{
    return m_high ; 
}
aiVector3D* AssimpSelection::getCenter()
{
    return m_center ; 
}
aiVector3D* AssimpSelection::getExtent()
{
    return m_extent ; 
}
aiVector3D* AssimpSelection::getUp()
{
    return m_up ; 
}


