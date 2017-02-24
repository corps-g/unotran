program test_angle

use angle

implicit none

integer :: l

l = 7
call initialize_angle(l, GL)
print *, "mu=", mu
print *, "wt=", wt
call finalize_angle()

call initialize_angle(l, DGL)
print *, "mu=", mu
print *, "wt=", wt
call finalize_angle()

end program test_angle
