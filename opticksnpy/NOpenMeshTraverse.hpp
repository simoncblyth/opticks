#pragma once

template <typename T>
struct NOpenMeshTraverse 
{
    typedef typename T::FaceHandle FH ; 
    typedef typename std::deque<FH>::iterator DFHI ;  
    typedef typename T::ConstFaceFaceIter    CFFI ;  

    NOpenMeshTraverse( const T& mesh, const NOpenMeshFind<T>& find, std::vector<FH>& faces, const FH seed, int verbosity  ) 
        :
        mesh(mesh),
        find(find),
        q(faces.begin(), faces.end()),
        steps(0),
        since(0),
        offset(0),
        verbosity(verbosity)
    {
        std::cout << " NOpenMeshTraverse seed : " << find.desc_face(seed) << std::endl ; 

        add(seed);
        collect();
        reorder(faces);
    }

    void reorder(std::vector<FH>& faces )
    {
        //assert( c.size() == faces.size() );   
        faces.assign( c.rbegin(), c.rend() );
    }

    void add(const FH fh)
    {
        _add(fh);
        //_add_buddies(fh);
    }

    void _add(const FH fh)
    {
        // if in q, add to contiguous and remove from q
        DFHI qi = std::find(q.begin(), q.end(), fh ) ; 
        DFHI ci = std::find(c.begin(), c.end(), fh ) ; 

        bool c_yes = ci != c.end() ;
        bool q_yes = qi != q.end() ;
        bool proceed = !c_yes && q_yes ;

        std::cout << "_add " << find.desc_face(fh) 
                      << " q " << q.size()
                      << " c " << c.size()
                      << ( c_yes ? " c_yes " : " " ) 
                      << ( q_yes ? " q_yes " : " " ) 
                      << ( proceed ? " proceed " : " " ) 
                      << std::endl ; 
 
        if( proceed  ) 
        {
            c.push_back(fh) ;  
            q.erase(qi);

       }
    }

    void _add_buddies(const FH fh)
    {
        // collect contiguous buddies of fh that are in the queue, and de-queue them
        for (CFFI cffi = mesh.cff_iter(fh); cffi.is_valid(); ++cffi) 
        {
            const FH buddy = *cffi ; 
            _add(buddy);
        }
    }

    void collect()
    {
        while(!q.empty() && steps < 100)
        {
            FH tip = c[c.size() - 1 - offset] ; 
            FH can = q.back(); q.pop_back() ; 

            bool is_contiguous = find.are_contiguous(tip, can) ;
            bool is_stuck = since > q.size() ; 

            if(verbosity > 1)
            std::cout
                << " q " << std::setw(3) << q.size()
                << " c " << std::setw(3) << c.size()
                << " since " << std::setw(3) << since
                << " o " << std::setw(3) << offset
                << "      "
                << " tip " << find.desc_face(tip)  
                << "      "
                << " can " << find.desc_face(can) 
                << "      "
                << ( is_contiguous ? " CONTIGUOUS " : "" )
                << ( is_stuck ? " STUCK " : "" )
                << std::endl 
                ;

            if(is_contiguous)
            {
                add(can);
                since = 0 ; 
                offset = 0 ; 
            }
            else
            {
                q.push_front(can);
                since++ ;
            }

            if(is_stuck && offset + 1 < c.size() )
            {
                offset++ ; 
                since = 0 ; 
            }
            steps++ ; 
        }
    }

    const T& mesh ; 
    const NOpenMeshFind<T>& find ; 

    std::deque<FH> q ; 
    std::deque<FH> c ;  

    unsigned steps ; 
    unsigned since ; 
    unsigned offset ; 

    int verbosity ; 

};



