#pragma once
#include <vector>
#include <string>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/*

OpticksQuery
==============


Allows selections of volumes of the scene hierarachy.
Query string examples from opticks/bin/op.sh::

    309 op-geometry-query-dyb()
    310 {
    311     case $1 in
    312    DYB|DLIN)  echo "range:3153:12221"  ;;
    313        DFAR)  echo "range:4686:18894"   ;;  #  
    314        IDYB)  echo "range:3158:3160" ;;  # 2 volumes : pvIAV and pvGDS
    315        JDYB)  echo "range:3158:3159" ;;  # 1 volume : pvIAV
    316        KDYB)  echo "range:3159:3160" ;;  # 1 volume : pvGDS
    317        LDYB)  echo "range:3156:3157" ;;  # 1 volume : pvOAV
    318        MDYB)  echo "range:3201:3202,range:3153:3154"  ;;  # 2 volumes : first pmt-hemi-cathode and ADE  
    319        DSST2)  echo "range:3155:3156,range:4440:4448" ;;    # large BBox discrep
    320        DLV17)  echo "range:3155:3156,range:2436:2437" ;;    #
    321        DLV30)  echo "range:3155:3156,range:3167:3168" ;;    #


*range*
     simple volume index range specified in the style of python slices
     (ie one past the last) can use multiple

*index*
     pick a start volume from which to recurse down from to
     a absolute max depth specified separately 

*depth*
     *depth* must be used in conjunction with *index*, it 
     specifies the absolute max depth to descend to, a non-zero query depth must be 
     specified to enable the recursive depth selection  

*name*
     selection based in name argument passed to OpticksQuery::selected

*merge*
     appears to not be completely implemented
     

*/


class OKCORE_API OpticksQuery {

public:
    typedef enum { UNDEFINED, NAME, INDEX, MERGE, DEPTH, RANGE, LVR } OpticksQuery_t ;

    static const char* UNDEFINED_ ; 
    static const char* NAME_ ; 
    static const char* INDEX_ ; 
    static const char* MERGE_ ; 
    static const char* DEPTH_ ; 
    static const char* RANGE_ ; 
    static const char* LVR_ ; 
    static const char* ALL_ ; 
    static const char* EMPTY_ ; 

    const char* getQueryTypeString() const ;
    OpticksQuery_t getQueryType() const ;

    void dump(const char* msg="OpticksQuery::dump") const ; 
    std::string desc() const ;

public:
    OpticksQuery(const char* query);
    bool selected(const char* name, unsigned int index, unsigned int depth, bool& recursive_select, unsigned lvIdx=0 );
private:
    void init();
public:
    // inline getters
    const char*  getQueryString() const ;
    const char*  getQueryDigest() const ;
    const char*  getQueryName() const ;
    int          getQueryIndex() const ;
    int          getQueryMerge() const ;
    int          getQueryDepth() const ;
    bool         isNoSelection() const ;
    bool         isFlatSelection() const ;
    std::vector<unsigned> getQueryRange() const ;
    std::vector<unsigned> getQueryLVRange() const ;
private:
    void parseQueryElement(const char* query);
    void parseQuery(const char* query);
private:
    const char*  m_query_string ; 
    const char*  m_query_digest ; 
    const char*  m_query_name ;
    unsigned int m_query_index ; 
    unsigned int m_query_merge ; 
    unsigned int m_query_depth ; 
    OpticksQuery_t m_query_type ; 

    std::vector<unsigned> m_query_range ; 
    std::vector<unsigned> m_query_lvrange ; 
    bool m_flat_selection ; 
    bool m_no_selection ; 

};

#include "OKCORE_TAIL.hh"


