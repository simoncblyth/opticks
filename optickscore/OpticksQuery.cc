#include "OpticksQuery.hh"

#include <algorithm>
#include <cstdio>

// npy-
#include "stringutil.hpp"
#include "NLog.hpp"


const char* OpticksQuery::UNDEFINED_ = "undefined" ; 
const char* OpticksQuery::NAME_ = "name" ; 
const char* OpticksQuery::INDEX_ = "index" ; 
const char* OpticksQuery::MERGE_ = "merge" ; 
const char* OpticksQuery::DEPTH_ = "depth" ; 
const char* OpticksQuery::RANGE_ = "range" ; 

void OpticksQuery::init()
{
    parseQuery(m_query_string);    
    dumpQuery("OpticksQuery::init dumpQuery");    

    LOG(info) << "OpticksQuery::OpticksQuery"
              << " m_query_string " << m_query_string 
              << " m_query_name " <<  (m_query_name ? m_query_name : "NULL" )
              << " m_query_index " <<  m_query_index 
              ;
}

void OpticksQuery::parseQuery(const char* query)
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

   LOG(info) << "OpticksQuery::parseQuery" 
             << " query:[" << query << "]"
             << " elements:" << elem.size()  
             << " queryType:" << getQueryTypeString()
             ;

}

void OpticksQuery::dumpQuery(const char* msg)
{
    LOG(info) << msg 
              << " queryType " << getQueryTypeString() ;

   if(m_query_type == RANGE)
   {
       size_t nrange = m_query_range.size() ;
       std::cerr << " nrange " << nrange ;
       for(unsigned int i=0 ; i < nrange ; i++)
       {
          std::cerr << " : " << m_query_range[i] ;
       }
       std::cerr << std::endl ; 
   } 
           
}

OpticksQuery::OpticksQuery_t OpticksQuery::getQueryType()
{
    return m_query_type ; 
}

const char* OpticksQuery::getQueryTypeString()
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



void OpticksQuery::parseQueryElement(const char* query)
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


