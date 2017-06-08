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
        nface(faces.size()),
        steps(0),
        miss(0),
        offset(0),
        verbosity(verbosity),
        maxstep(10000),
        ring1(false)
    {

        if(verbosity > 0)
        std::cout << " NOpenMeshTraverse::NOpenMeshTraverse START"
                  << " verbosity " << verbosity 
                  << " seed " << find.desc_face(seed)
                  << " steps " << steps
                  << " maxstep " << maxstep
                  << " nface " << nface
                  << std::endl ; 

        bool unq = true ; 
        _add_force(seed, unq );

        collect();
        reorder(faces);

        if(verbosity > 0)
        std::cout << " NOpenMeshTraverse::NOpenMeshTraverse DONE"
                  << " verbosity " << verbosity 
                  << std::endl ; 

    }

    void collect()
    {
        while(!q.empty() && steps < maxstep)
        {
            examine_next() ;
        }

        if(steps >= maxstep)
            LOG(fatal) << "NOpenMeshTraverse::collect"
                       << " verbosity " << verbosity 
                       << " steps " << steps 
                       << " maxstep " << maxstep
                       ; 
 
        assert( steps < maxstep );
    }

    void reorder(std::vector<FH>& faces )
    {
        if(c.size() != faces.size() ) std::cout << "NOpenMeshTraverse::reorder MISSING FACES " << std::endl ; 
        assert( c.size() == faces.size() );   
        faces.assign( c.rbegin(), c.rend() );
    }

    void examine_next()
    {
        FH can = q.back(); 
        q.pop_back() ; 

        _add_popped(can); // either succeeds into c, or is placed back in q 

        assert( q.size() + c.size() == nface );

        if(ring1)  // not sure if this helps
        {
            _add_ring1(can) ; 
            assert( q.size() + c.size() == nface );
        }

        bool is_stuck = miss > q.size() ; 
        if(is_stuck && offset + 1 < c.size() )
        {
            offset++ ; 
            tip = c[c.size() - 1 - offset] ;
            miss = 0 ; 

            if(verbosity > 3)
            std::cout << "unstucking " 
                      << " offset " << offset
                      << " tip " << find.desc_face(tip)
                      << std::endl ; 



        }
        steps++ ; 
    }

    void _add_ring1(const FH can)
    {
        bool c_prior_check = true ;  
        bool contiguous_check = false ;  // this is guaranteed
        bool unqueue = true ;  

        for (CFFI cffi = mesh.cff_iter(can); cffi.is_valid(); ++cffi) 
        {
            const FH buddy = *cffi ; 
            if(is_in_q(buddy))   // buddies not from q, so need to check
            {
                _add(buddy, c_prior_check, contiguous_check, unqueue ) ;  
            }
        }
    }

    void _add_popped(const FH can)
    {
        bool c_prior_check = false ; 
        bool contiguous_check = true ; 
        bool unqueue = false ;   // pop_back did this already 

        _add(can, c_prior_check, contiguous_check, unqueue ) ;  
    }


    bool is_in_q( const FH fh)
    {
        return std::find(q.begin(), q.end(), fh ) != q.end() ;
    }
    bool is_in_c( const FH fh)
    {
        return std::find(c.begin(), c.end(), fh ) != c.end() ;
    }
    bool is_connected_to_tip( const FH fh)
    {
        return find.are_contiguous(tip, fh) ;
    }


    void _add(const FH can, bool c_prior_check, bool contiguous_check, bool unqueue )
    {
        bool is_contiguous = contiguous_check ? is_connected_to_tip(can) : true ; 

        bool already = c_prior_check ? is_in_c(can)  : false ;  

        bool proceed = is_contiguous && !already ; 
 
        if( proceed ) 
        {
            _add_force( can, unqueue );
        }
        else
        {
            q.push_front(can);
            miss++ ;
        }

        if(verbosity > 3 || (verbosity > 2 && steps % 100) == 0)
        std::cout << "_add " << find.desc_face(can) 
                      << " verbosity " << std::setw(2) << verbosity
                      << " st " << std::setw(7) << steps
                      << " q " << std::setw(3) << q.size()
                      << " c " << std::setw(3) << c.size()
                      << " q+c " << std::setw(3) << q.size() + c.size()
                      << " m " << std::setw(3) << miss
                      << " o " << std::setw(3) << offset
                      << ( proceed ? " PROCEED " : " " ) 
                      << std::endl ; 

    }


    void _add_force(const FH fh, bool unq)
    {
        c.push_back(fh) ;  

        if(unq)
        {
            DFHI qi = std::find(q.begin(), q.end(), fh ) ; 
            assert( qi != q.end() );
            q.erase(qi);
        }

        assert( c.size() + q.size() == nface );

        // successful add changes tip
        tip = fh  ;
        miss = 0 ; 
        offset = 0 ; 
    }



    const T& mesh ; 
    const NOpenMeshFind<T>& find ; 

    std::deque<FH> q ; 
    std::deque<FH> c ;  

    FH tip ; 

    unsigned nface ; 
    unsigned steps ; 
    unsigned miss ; 
    unsigned offset ; 
    int      verbosity ; 
    unsigned maxstep ; 
    bool ring1 ; 


};



