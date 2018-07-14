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

subroutine array_addition(a,b,c,xs,ys) ! put them at the end because they will be optioinal anyways
  implicit none
  integer, intent(in) :: xs, ys ! sizes
  complex(8), dimension(xs,ys), intent(in) :: a,b
  complex(8), dimension(xs,ys), intent(inout) :: c
  c=a+b
  return
end subroutine array_addition

subroutine dlapack_mul(a,b,c,rows_a,columns_a,columns_b)
  implicit none
  integer, intent(in)                                     :: rows_a,columns_a,columns_b
  real(8), dimension(rows_a,columns_a), intent(in)     :: a
  real(8), dimension(columns_a, columns_b), intent(in) :: b
  real(8), dimension(rows_a, columns_b), intent(inout) :: c
  call dgemm('n', 'n', rows_a, columns_b, columns_a, 1.d0, a, rows_a, b, columns_a, 0.d0, c, rows_a)
  return
end subroutine

subroutine zlapack_mul(a,b,c,rows_a,columns_a,columns_b)
  implicit none
  integer, intent(in)                                     :: rows_a,columns_a,columns_b
  complex(kind=8), dimension(rows_a,columns_a), intent(in)     :: a
  complex(kind=8), dimension(columns_a, columns_b), intent(in) :: b
  complex(kind=8), dimension(rows_a, columns_b), intent(inout) :: c
  call zgemm('n', 'n', rows_a, columns_b, columns_a, 1.d0, a, rows_a, b, columns_a, 0.d0, c, rows_a)
  return
end subroutine

subroutine inverse_matrix_z(M, ierr, ndim)
  implicit none
  integer, intent(in)                                   :: ndim
  integer, intent(out)                                  :: ierr
  complex(kind=8), dimension(ndim,ndim), intent (inout) :: M
  integer                                               :: lwork
  integer,        allocatable                           :: ipiv(:)
  complex(kind=8), allocatable                          :: work(:)
  complex(kind=8)                                       :: work_query(1)
  allocate(ipiv(ndim))
  call zgetrf(ndim,ndim,M,ndim,ipiv,ierr)
  if (ierr .ne. 0) return
  call zgetri(ndim,M,ndim,ipiv,work_query,-1,ierr) ! query for optimal work space
  if (ierr .ne. 0) return
  lwork = int(work_query(1))
  allocate(work(lwork))
  call zgetri(ndim,M,ndim,ipiv,work,lwork,ierr)
  if (ierr .ne. 0) return
  deallocate(ipiv,work)
end subroutine
