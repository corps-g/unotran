module material
  implicit none
  integer :: nMaterials, nGroups, debugFlag
  double precision, allocatable, dimension(:) :: ebounds, velocity
  double precision, allocatable, dimension(:,:) :: sig_t, sig_f, vsig_f, chi
  double precision, allocatable, dimension(:,:,:,:) :: sig_s
    
  contains

  subroutine create(fileName)
    character(len=*), intent(in) :: fileName
    character(1000) :: materialName
    integer :: mat, group, groupp, L, nLegendre, dataPresent
    double precision :: t, f, vf, c, energyFission, energyCapture, gramAtomWeight
    double precision, allocatable, dimension(:) :: array1
    
    ! Read the file parameters
    open(unit=5, file=fileName)
    read(5,*) nMaterials, nGroups, debugFlag
    allocate(ebounds(nGroups+1))
    read(5,*) ebounds
    allocate(velocity(nGroups))
    read(5,*) velocity
    read(5,'(a)') materialName
    read(5,*) nLegendre, dataPresent, energyFission, energyCapture, gramAtomWeight
    
    ! Make space for cross sections
    allocate(sig_t(nMaterials, nGroups))
    allocate(sig_f(nMaterials, nGroups))
    allocate(vsig_f(nMaterials, nGroups))
    allocate(chi(nMaterials, nGroups))
    allocate(sig_s(nMaterials, nLegendre, nGroups, nGroups))
    allocate(array1(nGroups))
    
    ! Read the cross sections from the file
    do mat = 1, nMaterials
      if (mat > 1) then  ! The first material was read above to get array sizes
        read(5,'(a)') materialName
        read(5,*) nLegendre, dataPresent, energyFission, energyCapture, gramAtomWeight
      end if
      do group = 1, nGroups
        if (dataPresent == 1) then
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
      do L = 1, nLegendre
        do group = 1, nGroups
          read(5,*) array1
          do groupp = 1, nGroups
            sig_s(mat, L, group, groupp) = array1(groupp)
          end do
        end do
      end do
    end do
      
    
  end subroutine create

END module material
    

