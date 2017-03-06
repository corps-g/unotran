module material
  implicit none
  integer :: number_materials, number_groups, debugFlag, number_legendre
  double precision, allocatable, dimension(:) :: ebounds, velocity
  double precision, allocatable, dimension(:,:) :: sig_t, sig_f, vsig_f, chi
  double precision, allocatable, dimension(:,:,:,:) :: sig_s
  save
  contains

  subroutine create_material(fileName)
    character(len=*), intent(in) :: fileName
    character(1000) :: materialName
    integer :: mat, group, groupp, L, dataPresent
    double precision :: t, f, vf, c, energyFission, energyCapture, gramAtomWeight
    double precision, allocatable, dimension(:) :: array1
    
    ! Read the file parameters
    open(unit=5, file=fileName)
    read(5,*) number_materials, number_groups, debugFlag
    allocate(ebounds(number_groups + 1))
    read(5,*) ebounds
    allocate(velocity(number_groups))
    read(5,*) velocity
    read(5,'(a)') materialName
    read(5,*) number_legendre, dataPresent, energyFission, energyCapture, gramAtomWeight
    ! Count the highest order + zeroth order
    number_legendre = number_legendre - 1
    
    ! Make space for cross sections
    allocate(sig_t(number_materials, number_groups))
    allocate(sig_f(number_materials, number_groups))
    allocate(vsig_f(number_materials, number_groups))
    allocate(chi(number_materials, number_groups))
    allocate(sig_s(number_materials, 0:number_legendre, number_groups, number_groups))
    allocate(array1(number_groups))
    
    ! Read the cross sections from the file
    do mat = 1, number_materials
      if (mat .gt. 1) then  ! The first material was read above to get array sizes
        read(5,'(a)') materialName
        read(5,*) number_legendre, dataPresent, energyFission, energyCapture, gramAtomWeight
        ! Count the highest order + zeroth order
        number_legendre = number_legendre - 1
      end if
      do group = 1, number_groups
        if (dataPresent .eq. 1) then
          ! Read total and fission cross sections
          read(5,*) t, f, vf, c
          sig_t(mat, group) = t
          sig_f(mat, group) = f
          vsig_f(mat, group) = vf
          chi(mat, group) = c
        else
          ! Read just the total cross section
          read(5,*) t
          sig_t(mat, group) = t
          sig_f(mat, group) = 0.0
          vsig_f(mat, group) = 0.0
          chi(mat, group) = 0.0
        end if
      end do
      ! Read scattering cross section
      do L = 0, number_legendre
        do group = 1, number_groups
          read(5,*) array1
          do groupp = 1, number_groups
            sig_s(mat, L, group, groupp) = array1(groupp)
          end do
        end do
      end do
    end do
      
    
  end subroutine create_material

end module material
    

