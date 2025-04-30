! https://numpy.org/doc/2.2/f2py/f2py-examples.html
! pass array of strings from C to fortran
! this is used from ana/qcf.py see ana/qcf_ab.sh

subroutine find_first(needle, haystack, haystack_length, idx)
   !! Find the first index of `needle` in `haystack`.
   implicit none
   character(len=96), intent(in)  :: needle
   integer,           intent(in)  :: haystack_length
   character(len=96), intent(in), dimension(haystack_length) :: haystack 
!f2py intent(inplace) haystack
   integer, intent(out) :: idx
   integer :: k

   !print*,trim(needle)

   idx = -1
   do k = 1, haystack_length
       !print*,k,trim(haystack(k))
       if (haystack(k)==needle) then
            idx = k - 1
            exit
        endif
   enddo

end


subroutine foo( qu_len, qu, a_len,au,ax,an, b_len,bu,bx,bn, ab )
   implicit none

   integer, intent(in)  :: qu_len
   integer, intent(in)  :: a_len
   integer, intent(in)  :: b_len

   integer, intent(out),dimension(qu_len,3,2) :: ab(0:qu_len-1,0:2,0:1)

   integer :: ai
   integer :: bi
   integer :: k

   character(len=96), intent(in), dimension(qu_len) :: qu(0:qu_len-1)

   character(len=96), intent(in), dimension(a_len) :: au(0:a_len-1)
   integer,           intent(in), dimension(a_len) :: ax(0:a_len-1)
   integer,           intent(in), dimension(a_len) :: an(0:a_len-1)

   character(len=96), intent(in), dimension(b_len) :: bu(0:b_len-1)
   integer,           intent(in), dimension(b_len) :: bx(0:b_len-1)
   integer,           intent(in), dimension(b_len) :: bn(0:b_len-1)

   !print*,qu_len
   do k = 0, qu_len-1

       call find_first(qu(k), au, a_len, ai)
       call find_first(qu(k), bu, b_len, bi)

       !print*,k,qu(k)
       !print*,"ai", ai
       !print*,"bi", bi

       ab(k,0,0) = ai           ! ai is internal index into au, the A unique list 
       if (ai > -1) then
           ab(k,1,0) = ax(ai)   ! index of first occurrence in original A seq list
           ab(k,2,0) = an(ai)   ! count in A or 0 when not present 
       else
           ab(k,1,0) = -1 
           ab(k,2,0) = 0 
       end if

       ab(k,0,1) = bi           ! bi is internal index into bu, the B unique list 
       if (ai > -1) then
           ab(k,1,1) = bx(bi)   ! index of first occurrence in original B seq list
           ab(k,2,1) = bn(bi)   ! count in B or 0 when not present 
       else
           ab(k,1,1) = -1 
           ab(k,2,1) = 0 
       end if
   enddo

   !print*,a_len
   !do k = 1, a_len
   !    print*,k,ax(k),an(k),trim(au(k))
   !enddo

   !print*,b_len
   !do k = 1, b_len
   !    print*,k,bx(k),bn(k),trim(bu(k))
   !enddo

end


