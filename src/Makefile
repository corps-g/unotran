# Makefile for linear algebra tests 

#===============================================================================
# User Options
#===============================================================================

COMPILER = gfortran
OPTIMIZE = yes

#===============================================================================
# Vendor BLAS implementations
#===============================================================================

# For linking MKL, see:
#   http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
LIBMKLSERIAL   = -I$(MKLROOT)/include -Wl,--start-group       \
                 $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a   \
                 $(MKLROOT)/lib/intel64/libmkl_sequential.a   \
                 $(MKLROOT)/lib/intel64/libmkl_core.a         \
                 -Wl,--end-group -lpthread -lm
LIBMKLTHREAD   = -I$(MKLROOT)/include  -Wl,--start-group      \
                 $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a   \
                 $(MKLROOT)/lib/intel64/libmkl_intel_thread.a \
                 $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,    \
                 --end-group -openmp -lpthread -lm

# BLAS defines a generic set of functions with vendor-specific
# implementations; here, we use the blas implementation from the 
# software repository.
LIBBLAS        = -L/usr/lib -lblas -llapack -ffree-line-length-none -fcheck=all -fdefault-double-8 -fdefault-real-8

#===============================================================================
# Compiler Definitions
#===============================================================================

ifeq ($(COMPILER),gfortran)
  F90 = gfortran
  OMPFLAG = -fopenmp
  ifeq ($(OPTIMIZE),yes)
    OPT = -O3
  endif
endif

ifeq ($(COMPILER),ifort)
  F90 = ifort
  OMPFLAG = -openmp
  ifeq ($(OPTIMIZE),yes)
    OPT = -O3
  endif
endif

#===============================================================================
# Targets
#===============================================================================

vpath test_%.f90 
BC := /usr/bin/bc

objects = material mesh angle sweeper state

all: $(objects) unotran

# Use the repository's BLAS implementation
angle: angle.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@ 
	
test_angle: control.f90 angle.f90 ./test/test_angle.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_angle $^ 
	@./test/test_angle
	
dgm: material.f90 mesh.f90 angle.f90 state.f90 dgm.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_dgm: control.f90 mesh.f90 material.f90 angle.f90 state.f90 sweeper.f90 dgm.f90 dgmsweeper.f90 dgmsolver.f90 ./test/test_dgm.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_dgm $^ 
	@./test/test_dgm
	
dgmsweeper: material.f90 mesh.f90 angle.f90 state.f90 dgm.f90 dgmsweeper.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_dgmsweeper: control.f90 mesh.f90 material.f90 angle.f90 state.f90 dgm.f90 dgmsweeper.f90 sweeper.f90 ./test/test_dgmsweeper.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_dgmsweeper $^ 
	@./test/test_dgmsweeper
	
dgmsolver: material.f90 mesh.f90 angle.f90 state.f90 dgm.f90 dgmsweeper.f90 dgmsolver.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@

test_dgmsolver: control.f90 mesh.f90 material.f90 angle.f90 state.f90 dgm.f90 dgmsweeper.f90 sweeper.f90 dgmsolver.f90 ./test/test_dgmsolver.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_dgmsolver $^ 
	@./test/test_dgmsolver
	
mesh: mesh.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@ 
	
test_mesh: control.f90 mesh.f90 ./test/test_mesh.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_mesh $^ 
	@./test/test_mesh
	
material: material.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_material: control.f90 material.f90 ./test/test_material.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_material $^ 
	@./test/test_material
	
state: material.f90 mesh.f90 angle.f90 state.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_state: control.f90 mesh.f90 material.f90 angle.f90 state.f90 ./test/test_state.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_state $^ 
	@./test/test_state
	
sweeper: material.f90 mesh.f90 angle.f90 state.f90 sweeper.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_sweeper: control.f90 mesh.f90 material.f90 angle.f90 state.f90 sweeper.f90 solver.f90 ./test/test_sweeper.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_sweeper $^ 
	@./test/test_sweeper

unotran: control.f90 material.f90 mesh.f90 angle.f90 state.f90 dgm.f90 sweeper.f90 dgmsweeper.f90 solver.f90 dgmsolver.f90 main.f90
	$(F90) $(OMPFLAG) $(LIBBLAS) -o xunotran $^ 
	
solver: material.f90 mesh.f90 angle.f90 state.f90 dgm.f90 sweeper.f90 dgmsweeper.f90 solver.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_solver: control.f90 mesh.f90 material.f90 angle.f90 state.f90 sweeper.f90 dgm.f90 dgmsweeper.f90 solver.f90 ./test/test_solver.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_solver $^ 
	@./test/test_solver
	
test: test_angle test_mesh test_material test_state test_sweeper test_solver test_dgm test_dgmsweeper test_dgmsolver

# Remove objects, module files, and exectables
clean:
	@rm -f *.o *.mod *-x *wrap.cxx xunotran

# Same, but leave the executable
neat:
	@rm -f *.o *.mod *-x *wrap.cxx

#===============================================================================
# Rules
#===============================================================================

.SUFFIXES: .F90 .o
.PHONY: clean neat

%.o: %.f90
	$(F90) $(F90FLAGS) $(INCLUDE) -c $<
