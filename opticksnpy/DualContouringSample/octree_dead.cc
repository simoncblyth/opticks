

      /* 
        nvec3 cpos = coarse.fpos(c); 
        int corners2 = field.zcorners(cpos, nominal.elem*subtile.size ) ; 
        if(corners != corners2)
            std::cout 
                 << " corners 0b" << std::bitset<8>(corners) 
                 << " corners2 0b" << std::bitset<8>(corners2)
                 << " cpos " << cpos.desc()
                 << " nominal.elem " << nominal.elem
                 << " subtile.size " << subtile.size
                 << std::endl ;  
        */
        //assert(corners == corners2);





bool HasChildren( const ivec3& min , const int nodeSize, std::function<float(float,float,float)>* f, const nvec4& ce)
{
	const int childSize = nodeSize / 2;
    if(childSize == 1) return HasLeaves( min, childSize, f, ce );
    for (int i = 0; i < 8; i++)
    {
         ivec3 child_min = min + (CHILD_MIN_OFFSETS[i] * childSize) ;
         bool has = HasChildren(child_min, childSize, f, ce ); 
         if(!has) continue ; 
         return true ; 
    }
    return false ; 
}



bool HasLeaves( const ivec3& min, const int leafSize, std::function<float(float,float,float)>* f, const nvec4& ce )
{
    assert(leafSize == 1);
    for (int i = 0; i < 8; i++)
    {
        ivec3 leaf_min = min + (CHILD_MIN_OFFSETS[i] * leafSize); 
        int corners = Corners(leaf_min, f, ce );  
        if(corners == 0 || corners == 255) continue ;  // not leaf, keep looking
        return true ;    // found one, so early exit 
    }
    return false ; 
}


OctreeNode* ConstructOctreeNodes_0(OctreeNode* node, std::function<float(float,float,float)>* f, const nvec4& ce)
{
	if (!node)
	{
		return nullptr;
	}

	if (node->size == 1)
	{
		return ConstructLeaf(node, f, ce);
	}
	
	const int childSize = node->size / 2;
	bool hasChildren = false;

	for (int i = 0; i < 8; i++)
	{
		OctreeNode* child = new OctreeNode;
		child->size = childSize;
		child->min = node->min + (CHILD_MIN_OFFSETS[i] * childSize);
		child->type = Node_Internal;

		node->children[i] = ConstructOctreeNodes_0(child, f, ce);
		hasChildren |= (node->children[i] != nullptr);
	}

	if (!hasChildren)
	{
		delete node;
		return nullptr;
	}

	return node;
}



