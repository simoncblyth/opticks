#pragma once

#include <assimp/types.h>
#include <vector>

class AssimpNode ;

class AssimpSelection {

public:
    AssimpSelection(AssimpNode* root, const char* query);
    virtual ~AssimpSelection();

private:
    void init();

public:
    unsigned int select(const char* query);
    void dumpSelection();
    unsigned int getNumSelected();
    AssimpNode* getSelectedNode(unsigned int i);
    bool isFlatSelection();
    bool isNoSelection();
    int getQueryMerge();
    int getQueryDepth();
    bool contains(AssimpNode* node);

private:
    void addToSelection(AssimpNode* node);
    void selectNodes(AssimpNode* node, unsigned int depth, bool rselect=false);
    void parseQueryElement(const char* query);
    void parseQuery(const char* query);

public:
    // bounds
    void dump();
    void bounds();
    aiVector3D* getLow();
    aiVector3D* getHigh();
    aiVector3D* getCenter();
    aiVector3D* getExtent();
    aiVector3D* getUp();

    void findBounds();
    void findBounds(AssimpNode* node, aiVector3D& low, aiVector3D& high );


private:
    char* m_query ; 

    char* m_query_name ;

    int m_query_index ; 

    int m_query_merge ; 

    int m_query_depth ; 

    std::vector<int> m_query_range ; 

    bool m_flat_selection ; 

    bool m_no_selection ; 

    std::vector<AssimpNode*> m_selection ; 

private:

   unsigned int m_count ; 

   unsigned int m_index ; 

   AssimpNode*  m_root ; 


private:

   aiVector3D* m_low ; 

   aiVector3D* m_high ; 

   aiVector3D* m_center ; 

   aiVector3D* m_extent ; 

   aiVector3D* m_up ; 



};




inline AssimpSelection::AssimpSelection(AssimpNode* root, const char* query) 
    : 
    m_count(0),
    m_index(0),
    m_root(root),    
    m_query(strdup(query)),
    m_query_name(NULL),
    m_query_index(0), 
    m_query_merge(0), 
    m_query_depth(0), 
    m_flat_selection(false),
    m_no_selection(false),
    m_low(NULL),
    m_high(NULL),
    m_center(NULL),
    m_extent(NULL),
    m_up(NULL)
{
    init();
}


inline int AssimpSelection::getQueryMerge()
{
    return m_query_merge ;  
}
inline int AssimpSelection::getQueryDepth()
{
    return m_query_depth == 0 ? 100 : m_query_depth ;  
}
inline bool AssimpSelection::isFlatSelection()
{
    return m_flat_selection ; 
}
inline bool AssimpSelection::isNoSelection()
{
    return m_no_selection ; 
}




