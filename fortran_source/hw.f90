real(8) function hw1(r1,r2)
  implicit none
  real(8) :: r1,r2
  hw1=(r1+r2)
  return
end function hw1

subroutine hw2(r1,r2,s)
  implicit none
  real(8), intent(in) :: r1,r2
  real(8), intent(out) :: s
  s=sin(r1+r2)
  return
end subroutine hw2

subroutine hw3(a,b,c,d)
  implicit none
  complex(8), intent(in) :: a,b
  complex(8), intent(out) :: c,d
  c = a+b
  d = a-b
  return
end subroutine hw3

subroutine array_addition(xs,ys,a,b,c)
  implicit none
  integer, intent(in) :: xs, ys ! sizes
  complex(8), dimension(xs,ys), intent(in) :: a,b
  complex(8), dimension(xs,ys), intent(inout) :: c
  integer i,j

  c=a+b
  return

end subroutine array_addition
