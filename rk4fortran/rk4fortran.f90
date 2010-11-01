subroutine rk4stepks(x, y, h, yout, n,m)
!   Do one step of the classical 4th order Runge Kutta method,
!   starting from y at x with time step h and derivatives given by derivs
  implicit none
  integer :: n,m
  real(8) :: x, h
  real(8) :: y(0:n-1,0:m-1)
  real(8) :: yout(0:n-1,0:m-1)

!f2py real(8), intent(in) :: x, h
!f2py real(8), intent(out), dimension(0:n-1,0:m-1) :: yout
!f2py real(8), intent(in), dimension(0:n-1,0:m-1) :: y
!f2py integer, intent(in) :: n,m


  yout = y
    return
    end subroutine
