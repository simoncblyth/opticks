#pragma once
#include <vector>
#include <string>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksQuery {

public:
    typedef enum { UNDEFINED, NAME, INDEX, MERGE, DEPTH, RANGE } OpticksQuery_t ;

    static const char* UNDEFINED_ ; 
    static const char* NAME_ ; 
    static const char* INDEX_ ; 
    static const char* MERGE_ ; 
    static const char* DEPTH_ ; 
    static const char* RANGE_ ; 

    const char* getQueryTypeString();
    OpticksQuery_t getQueryType();
    void dumpQuery(const char* msg="OpticksQuery::dumpQuery");
    std::string description();

public:
    OpticksQuery(const char* query);
    bool selected(const char* name, unsigned int index, unsigned int depth, bool& recursive_select );
private:
    void init();
public:
    // inline getters
    const char* getQueryString();
    const char* getQueryName();
    int getQueryIndex();
    int getQueryMerge();
    int getQueryDepth();
    std::vector<unsigned int> getQueryRange();
    bool isNoSelection();
    bool isFlatSelection();
private:
    void parseQueryElement(const char* query);
    void parseQuery(const char* query);
private:
    const char*  m_query_string ; 
    const char*  m_query_name ;
    unsigned int m_query_index ; 
    unsigned int m_query_merge ; 
    unsigned int m_query_depth ; 
    OpticksQuery_t m_query_type ; 

    std::vector<unsigned int> m_query_range ; 
    bool m_flat_selection ; 
    bool m_no_selection ; 

};

#include "OKCORE_TAIL.hh"


