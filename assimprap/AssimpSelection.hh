#pragma once

#include <assimp/types.h>
#include <vector>

class AssimpNode ;
class OpticksQuery ; 

#include "ASIRAP_API_EXPORT.hh"
#include "ASIRAP_HEAD.hh"

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


