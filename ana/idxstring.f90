
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

  




