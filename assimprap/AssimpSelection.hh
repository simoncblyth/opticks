#pragma once

#include <assimp/types.h>
#include <vector>

class AssimpNode ;
class OpticksQuery ; 

#include "ASIRAP_API_EXPORT.hh"
#include "ASIRAP_HEAD.hh"

/*
AssimpSelection
==================

Contains m_selection vector of AssimpNode.

Canonically instanciated from AssimpGGeo::load and used
to control the assimp to GGeo conversion.
The selection details, such as volume ranges are 
passed in via the OpticksQuery instance.

Node selection via OpticksQuery based upon 
full volume tree traversal index, name, depth
controls which AssimpNode get added to the m_selection.::

    126    const char* name = node->getName();
    127    unsigned int index = node->getIndex();
    ...
    139    bool selected = m_query->selected(name, index, depth, recursive_select);


Volume range selection is the most commonly used one::

    206 bool OpticksQuery::selected(const char* name, unsigned int index, unsigned int depth, bool& recursive_select )
    ...
    237    else if(m_query_range.size() > 0)
    238    {
    239        assert(m_query_range.size() % 2 == 0);
    240        for(unsigned int i=0 ; i < m_query_range.size()/2 ; i++ )
    241        {
    242            if( index >= m_query_range[i*2+0] && index < m_query_range[i*2+1] ) _selected = true ;
    243        }
    ...

*/

class ASIRAP_API AssimpSelection {
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
    void dump(const char* msg="AssimpSelection::dump");
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

#include "ASIRAP_TAIL.hh"


