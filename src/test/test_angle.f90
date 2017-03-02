program test_angle

  use angle

  implicit none

  integer :: l = 7, t1=1, t2=1
  double precision :: mu_test(7), wt_test(7)

  mu_test = [0.9862838087,0.9284348837,0.8272013151,0.6872929048,0.5152486364,0.3191123689,0.1080549487]
  wt_test = [0.0351194603,0.0801580872,0.1215185707,0.1572031672,0.1855383975,0.2051984637,0.2152638535]

  call initialize_angle(l, GL)
  if (norm2(mu-mu_test) .lt. 1e-7) then
    write(*,"(A)",advance="no") '.'
  else
    write(*,"(A)",advance="no") 'F'
    t1=0
  end if
  call finalize_angle()

  mu_test = [0.9745539562,0.8707655928,0.7029225757,0.5000000000,0.2970774243,0.1292344072,0.0254460438]
  wt_test = [0.0647424831,0.1398526957,0.1909150253,0.2089795918,0.1909150253,0.1398526957,0.0647424831]
  
  call initialize_angle(l, DGL)
  if (norm2(mu-mu_test) .lt. 1e-7) then
    write(*,"(A)",advance="no") '.'
  else
    write(*,"(A)",advance="no") 'F'
    t2=0
  end if
  call finalize_angle()
  
  if (t1 .eq. 0) then
    print *, 'angle: GL option failed'
  else if (t2 .eq. 0) then
    print *, 'angle: DGL option failed'
  else
    print *
  end if

end program test_angle
