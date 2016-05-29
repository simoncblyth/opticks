#pragma once

#include <assimp/types.h>
#include <vector>

class AssimpNode ;
class OpticksQuery ; 

class AssimpSelection {
public:
    AssimpSelection(AssimpNode* root, OpticksQuery* query);
private:
    void init();
public:
    unsigned int select(const char* query);
    void dumpSelection();
    unsigned int getNumSelected();
    AssimpNode* getSelectedNode(unsigned int i);
    bool contains(AssimpNode* node);
private:
    void addToSelection(AssimpNode* node);
    void selectNodes(AssimpNode* node, unsigned int depth, bool rselect=false);
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
    OpticksQuery* m_query ; 

    const char* m_query_string ; 
    const char* m_query_name ;
    unsigned int m_query_index ; 
    unsigned int m_query_merge ; 
    unsigned int m_query_depth ; 
//    AssimpQuery_t m_query_type ; 
    bool m_flat_selection ; 
    bool m_no_selection ; 
    std::vector<unsigned int> m_query_range ; 

private:
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

inline AssimpSelection::AssimpSelection(AssimpNode* root, OpticksQuery* query) 
    : 
    m_query(query),
    m_query_string(NULL),
    m_query_name(NULL),
    m_query_index(0), 
    m_query_merge(0), 
    m_query_depth(0), 
//    m_query_type(UNDEFINED), 
    m_flat_selection(false),
    m_no_selection(false),

    m_count(0),
    m_index(0),
    m_root(root),    

    m_low(NULL),
    m_high(NULL),
    m_center(NULL),
    m_extent(NULL),
    m_up(NULL)
{
    init();
}



inline unsigned int AssimpSelection::getNumSelected()
{
    return m_selection.size();
}

inline AssimpNode* AssimpSelection::getSelectedNode(unsigned int i)
{
    return i < m_selection.size() ? m_selection[i] : NULL ; 
}

inline void AssimpSelection::addToSelection(AssimpNode* node)
{
     m_selection.push_back(node);
}



