program test_angle

  use angle

  implicit none

  integer :: l = 7
  double precision :: mu_test(7), wt_test(7)

  mu_test = [0.9862838087,0.9284348837,0.8272013151,0.6872929048,0.5152486364,0.3191123689,0.1080549487]
  wt_test = [0.0351194603,0.0801580872,0.1215185707,0.1572031672,0.1855383975,0.2051984637,0.2152638535]

  call initialize_angle(l, GL)
  if (norm2(mu-mu_test) .lt. 1e-7) then
    print *, '.'
  else
    print *, 'angle: GL option failed'
  end if
  call finalize_angle()

  call initialize_angle(l, DGL)
  print *, "mu=", mu
  print *, "wt=", wt
  call finalize_angle()

end program test_angle
