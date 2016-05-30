#pragma once
#include <vector>
#include <string>

class OpticksQuery {

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

inline OpticksQuery::OpticksQuery(const char* query) 
    : 
    m_query_string(strdup(query)),
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

inline const char* OpticksQuery::getQueryString()
{
    return m_query_string ; 
}
inline const char* OpticksQuery::getQueryName()
{
    return m_query_name ; 
}
inline int OpticksQuery::getQueryIndex()
{
    return m_query_index ;  
}
inline int OpticksQuery::getQueryMerge()
{
    return m_query_merge ;  
}
inline int OpticksQuery::getQueryDepth()
{
    return m_query_depth == 0 ? 100 : m_query_depth ;  
}
inline std::vector<unsigned int> OpticksQuery::getQueryRange()
{
    return m_query_range ;  
}

inline bool OpticksQuery::isFlatSelection()
{
    return m_flat_selection ; 
}
inline bool OpticksQuery::isNoSelection()
{
    return m_no_selection ; 
}


