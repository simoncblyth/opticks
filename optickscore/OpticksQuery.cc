#include <algorithm>
#include <cstdio>
#include <sstream>
#include <cstring>

// brap-
#include "BStr.hh"
#include "SDigest.hh"
#include "PLOG.hh"

// okc-
#include "OpticksQuery.hh"

const char* OpticksQuery::UNDEFINED_ = "undefined" ; 
const char* OpticksQuery::NAME_ = "name" ; 
const char* OpticksQuery::INDEX_ = "index" ; 
const char* OpticksQuery::MERGE_ = "merge" ; 
const char* OpticksQuery::DEPTH_ = "depth" ; 
const char* OpticksQuery::RANGE_ = "range" ; 
const char* OpticksQuery::LVR_ = "lvr" ; 
const char* OpticksQuery::ALL_   = "all" ; 
const char* OpticksQuery::EMPTY_   = "" ; 


OpticksQuery::OpticksQuery(const char* query) 
    : 
    m_query_string(strdup(query)),
    m_query_digest(SDigest::md5digest_( m_query_string, strlen(m_query_string))),
    m_query_name(NULL),
    m_query_index(0), 
    m_query_merge(0), 
    m_query_depth(0), 
    m_query_type(UNDEFINED), 
    m_flat_selection(false),
    m_no_selection(false)
{
    init();
}

const char* OpticksQuery::getQueryString() const 
{
    return m_query_string ; 
}
const char* OpticksQuery::getQueryDigest() const 
{
    return m_query_digest ; 
}

const char* OpticksQuery::getQueryName() const
{
    return m_query_name ; 
}
int OpticksQuery::getQueryIndex() const 
{
    return m_query_index ;  
}
int OpticksQuery::getQueryMerge() const
{
    return m_query_merge ;  
}
int OpticksQuery::getQueryDepth() const 
{
    return m_query_depth == 0 ? 100 : m_query_depth ;  
}
std::vector<unsigned> OpticksQuery::getQueryRange() const 
{
    return m_query_range ;  
}
std::vector<unsigned> OpticksQuery::getQueryLVRange() const 
{
    return m_query_lvrange ;  
}


bool OpticksQuery::isFlatSelection() const 
{
    return m_flat_selection ; 
}
bool OpticksQuery::isNoSelection() const 
{
    return m_no_selection ; 
}


void OpticksQuery::init()
{
    parseQuery(m_query_string);    
    //dump("OpticksQuery::init");    
}

void OpticksQuery::dump(const char* msg) const 
{
    LOG(info) << msg << desc() ; 
}

std::string OpticksQuery::desc() const 
{
   std::stringstream ss ;  
   ss 
      << " queryType " << getQueryTypeString() 
      << " query_string " << m_query_string 
      << " query_name " <<  (m_query_name ? m_query_name : "NULL" )
      << " query_index " <<  m_query_index 
      << " query_depth " <<  m_query_depth
      << " no_selection " <<  m_no_selection
      ;

   if(m_query_type == RANGE)
   {
       size_t num_range = m_query_range.size() ;
       ss << " num_range " << num_range ;
       for(unsigned i=0 ; i < num_range ; i++)
       {
          ss << " : " << m_query_range[i] ;
       }
   } 
   else if(m_query_type == LVR)
   {
       size_t num_lvrange = m_query_lvrange.size() ;
       ss << " num_lvrange " << num_lvrange ;
       for(unsigned i=0 ; i < num_lvrange ; i++)
       {
          ss << " : " << m_query_lvrange[i] ;
       }
 
   }

   return ss.str();
}

void OpticksQuery::parseQuery(const char* query)
{
   // "name:helo,index:yo,range:10"
   // split at "," and then extract values beyond the "token:" 

   if(strcmp(query,EMPTY_)==0) 
   { 
       m_no_selection = true ; 
   } 
   else if(strcmp(query,ALL_)==0) 
   {
       m_no_selection = true ; 
   }
   else
   {
       std::vector<std::string> elem ; 
       BStr::split(elem, query, ',');

       for(unsigned int i=0 ; i < elem.size() ; i++ ) parseQueryElement( elem[i].c_str() );

       LOG(trace) << "OpticksQuery::parseQuery" 
                 << " query:[" << query << "]"
                 << " elements:" << elem.size()  
                 << " queryType:" << getQueryTypeString()
                 ;
   }
}


OpticksQuery::OpticksQuery_t OpticksQuery::getQueryType() const 
{
    return m_query_type ; 
}

const char* OpticksQuery::getQueryTypeString() const 
{
   const char* type = UNDEFINED_ ;
   switch(m_query_type)
   {
      case UNDEFINED: type=UNDEFINED_ ;break; 
      case NAME     : type=NAME_      ;break; 
      case INDEX    : type=INDEX_     ;break; 
      case RANGE    : type=RANGE_     ;break; 
      case LVR      : type=LVR_       ;break; 
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
   const char* lvr_token = "lvr:" ;
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
       BStr::split(elem, query+strlen(range_token), ':'); 
       assert(elem.size() == 2);
       for(unsigned int i=0 ; i<elem.size() ; ++i)
       {
           int query_range_elem = atoi(elem[i].c_str());
           assert(query_range_elem > -1 );
           m_query_range.push_back(query_range_elem) ;
       }
       m_flat_selection = true ; 
   } 
   else if(strncmp(query,lvr_token, strlen(lvr_token)) == 0)
   {
       m_query_type = LVR ; 
       std::vector<std::string> elem ; 
       BStr::split(elem, query+strlen(lvr_token), ':'); 
       assert(elem.size() == 2);
       for(unsigned int i=0 ; i<elem.size() ; ++i)
       {
           int query_lvrange_elem = atoi(elem[i].c_str());
           assert(query_lvrange_elem > -1 );
           m_query_lvrange.push_back(query_lvrange_elem) ;
       }
       m_flat_selection = true ; 
  } 
}


/**
OpticksQuery::selected
------------------------

Subsequent calls to selected from a node traverse
need to supply a reference to the same recursive_select 
boolean. 


**/

bool OpticksQuery::selected(const char* name, unsigned int index, unsigned int depth, bool& recursive_select, unsigned lvIdx )
{
   bool _selected(false) ; 

   if(m_no_selection)
   {
       _selected = true ;
   }
   else if(m_query_name)
   {
       // startswith m_query_name ?
       if(strncmp(name,m_query_name,strlen(m_query_name)) == 0)
       {
           _selected = true ;
       }
   }
   else if (m_query_index != 0)
   {
       if( index == m_query_index )
       {
           _selected = true ;

           recursive_select = true ;

           // kick-off recursive select, note it then never 
           // gets turned off, but it only causes selection for depth within range 
       }
       else if ( recursive_select )
       {
           if( m_query_depth > 0 && depth < m_query_depth ) _selected = true ;
       }
   }
   else if(m_query_range.size() > 0)
   {
       assert(m_query_range.size() % 2 == 0);
       for(unsigned i=0 ; i < m_query_range.size()/2 ; i++ )
       {
           if( index >= m_query_range[i*2+0] && index < m_query_range[i*2+1] ) _selected = true ;
       }
   }
   else if(m_query_lvrange.size() > 0)
   {
       assert(m_query_lvrange.size() % 2 == 0);
       for(unsigned i=0 ; i < m_query_lvrange.size()/2 ; i++ )
       {
           if( lvIdx >= m_query_lvrange[i*2+0] && lvIdx < m_query_lvrange[i*2+1] ) _selected = true ;
       }
   }
   return _selected ; 
}


