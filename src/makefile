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
LIBBLAS        = -L/usr/lib -lblas -llapack

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

all: $(objects) main

# Use the repository's BLAS implementation
angle: angle.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@ 
	
test_angle: control.f90 angle.f90 ./test/test_angle.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_angle.so $^ 
	@./test/test_angle.so
	
dgm: material.f90 mesh.f90 angle.f90 state.f90 dgm.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_dgm: control.f90 mesh.f90 material.f90 angle.f90 state.f90 sweeper.f90 dgmsweeper.f90 dgmsolver.f90 dgm.f90 ./test/test_dgm.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_dgm.so $^ 
	@./test/test_dgm.so
	
dgmsweeper: material.f90 mesh.f90 angle.f90 state.f90 dgm.f90 dgmsweeper.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_dgmsweeper: control.f90 mesh.f90 material.f90 angle.f90 state.f90 dgm.f90 sweeper.f90 dgmsweeper.f90 dgmsolver.f90 ./test/test_dgmsweeper.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_dgmsweeper.so $^ 
	@./test/test_dgmsweeper.so
	
dgmsolver: material.f90 mesh.f90 angle.f90 state.f90 dgm.f90 dgmsweeper.f90 dgmsolver.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@

test_dgmsolver: control.f90 mesh.f90 material.f90 angle.f90 state.f90 dgm.f90 dgmsweeper.f90 sweeper.f90 dgmsolver.f90 ./test/test_dgmsolver.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_dgmsolver.so $^ 
	@./test/test_dgmsolver.so
	
mesh: mesh.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@ 
	
test_mesh: control.f90 mesh.f90 ./test/test_mesh.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_mesh.so $^ 
	@./test/test_mesh.so
	
material: material.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_material: control.f90 material.f90 ./test/test_material.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_material.so $^ 
	@./test/test_material.so
	
state: material.f90 mesh.f90 angle.f90 state.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_state: control.f90 mesh.f90 material.f90 angle.f90 state.f90 ./test/test_state.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_state.so $^ 
	@./test/test_state.so
	
sweeper: material.f90 mesh.f90 angle.f90 state.f90 sweeper.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_sweeper: control.f90 mesh.f90 material.f90 angle.f90 state.f90 sweeper.f90 ./test/test_sweeper.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_sweeper.so $^ 
	@./test/test_sweeper.so

main: control.f90 material.f90 mesh.f90 angle.f90 state.f90 dgm.f90 sweeper.f90 dgmsweeper.f90 solver.f90 dgmsolver.f90 main.f90
	$(F90) $(OMPFLAG) $(LIBBLAS) -o main.exe $^ 
	
solver: material.f90 mesh.f90 angle.f90 state.f90 dgm.f90 sweeper.f90 dgmsweeper.f90 solver.f90
	f2py --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_solver: control.f90 mesh.f90 material.f90 angle.f90 state.f90 sweeper.f90 dgm.f90 dgmsweeper.f90 solver.f90 ./test/test_solver.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_solver.so $^ 
	@./test/test_solver.so
	
test: test_angle test_mesh test_material test_state test_dgm test_dgmsweeper test_dgmsolver test_solver

# Remove objects, module files, and exectables
clean:
	@rm -f *.o *.mod *-x *wrap.cxx *.so test/*.so

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
