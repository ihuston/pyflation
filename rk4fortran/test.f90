! FILE: FIB1.F
      subroutine fib(A,N)
!
!     CALCULATE FIRST N FIBONACCI NUMBERS
!
      implicit none
      integer :: N, I
      REAL*8 :: A(N)
      DO I=1,N
         IF (I.EQ.1) THEN
            A(I) = 0.0D0
         ELSEIF (I.EQ.2) THEN
            A(I) = 1.0D0
         ELSE
            A(I) = A(I-1) + A(I-2)
         ENDIF
      ENDDO
      END

subroutine vec_in_out(in_matrix, in_out_vec, n ,m, p, out_vector)
  implicit none
  integer :: n,m,p
  real(8) :: in_matrix(n,m)
  real(8) :: out_vector(n+m-1)
  real(8) :: in_out_vec(p)

!f2py real(8), intent(out), dimension(n+m-1) :: out_matix
!f2py real(8), intent(in), dimension(n,m) :: in_matrix
!f2py real(8), intent(inout), dimension(p) :: in_out_vec
!f2py integer, intent(in) :: n,m, p


  out_vector = (/ in_matrix(:,1), in_matrix(:,2) /)
  in_out_vec = in_out_vec*5


end subroutine vec_in_out
