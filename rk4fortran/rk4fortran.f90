subroutine rk4stepks(x, y, out_vector, n,m,p, in_out_vec)
!   Do one step of the classical 4th order Runge Kutta method,
!   starting from y at x with time step h and derivatives given by derivs
  implicit none
  integer :: n,m,p
  real(8) :: x
  real(8) :: y(n,m)
  real(8) :: out_vector(n+m-1)
  real(8) :: in_out_vec(p)

!f2py real(8), intent(in) :: x
!f2py real(8), intent(out), dimension(n+m-1) :: out_matix
!f2py real(8), intent(in), dimension(n,m) :: y
!f2py real(8), intent(inout), dimension(p) :: in_out_vec
!f2py integer, intent(in) :: n,m, p


  out_vector = (/ y(:,1), y(:,2) /)
  in_out_vec = in_out_vec*5
    return
    end subroutine
