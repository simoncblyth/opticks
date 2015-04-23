#ifndef ASSIMPSELECTION_H
#define ASSIMPSELECTION_H

#include <assimp/types.h>
#include <vector>

class AssimpNode ;

class AssimpSelection {

public:
    AssimpSelection(AssimpNode* root, const char* query);
    virtual ~AssimpSelection();

public:
    unsigned int select(const char* query);
    void dumpSelection();
    unsigned int getNumSelected();
    AssimpNode* getSelectedNode(unsigned int i);
    bool isFlatSelection();
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

    bool m_is_flat_selection ; 

    std::vector<AssimpNode*> m_selection ; 

private:

   unsigned int m_index ; 

private:

   aiVector3D* m_low ; 

   aiVector3D* m_high ; 

   aiVector3D* m_center ; 

   aiVector3D* m_extent ; 

   aiVector3D* m_up ; 



};

#endif
