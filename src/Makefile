# Makefile for linear algebra tests 

#===============================================================================
# Compilation settings
#===============================================================================

F90 = gfortran
F90FLAGS = -O0 -fcheck=all -fdefault-double-8 -fdefault-real-8
LDFLAGS = 


#===============================================================================
# Object Files
#===============================================================================

objects =        \
angle.o          \
control.o        \
dgm.o            \
dgmsolver.o      \
dgmsweeper.o     \
material.o       \
mesh.o           \
solver.o         \
state.o          \
sweeper.o
 
#===============================================================================
# Targets
#===============================================================================

# main executable
xdgm: $(objects) main.f90
	$(F90) $(F90FLAGS) $(LDFLAGS) $^ -o $@ 

# executable tests
test_angle: pydgm
	python test/test_angle.py
test_dgm: pydgm
	python test/test_dgm.py
test_dgmsolver: pydgm
	python test/test_dgmsolver.py
test_dgmsweeper: pydgm
	python test/test_dgmsweeper.py
test_material: pydgm
	python test/test_material.py
test_mesh: pydgm
	python test/test_mesh.py
test_solver: pydgm
	python test/test_solver.py
test_state: pydgm
	python test/test_state.py
test_sweeper: pydgm
	python test/test_sweeper.py
test: pydgm
	python test/test_all.py

# f2py module
pydgm: control.f90 angle.f90 material.f90 mesh.f90 state.f90 sweeper.f90 solver.f90 dgm.f90 dgmsweeper.f90 dgmsolver.f90
	@rm -f pydgm*so
	f2py --f90flags="$(F90FLAGS)" -c -m $@ $^

# Remove objects, module files, and exectables
clean:
	@rm -f *.o *.mod pydgm*so xdgm test/*.x


#===============================================================================
# Rules
#===============================================================================

.SUFFIXES: .F90 .o
.PHONY: clean neat

%.o: %.f90
	$(F90) $(F90FLAGS) $(INCLUDE) -c $<


#===============================================================================
# Dependencies
#===============================================================================

angle.o: control.o
dgm.o: control.o
dgm.o: material.o
dgm.o: mesh.o
dgm.o: angle.o
dgm.o: state.o
dgmsolver.o: control.o
dgmsolver.o: material.o
dgmsolver.o: angle.o
dgmsolver.o: mesh.o
dgmsolver.o: state.o
dgmsolver.o: dgm.o
dgmsolver.o: dgmsweeper.o
dgmsweeper.o: control.o
dgmsweeper.o: material.o
dgmsweeper.o: mesh.o
dgmsweeper.o: angle.o
dgmsweeper.o: sweeper.o
dgmsweeper.o: state.o
dgmsweeper.o: dgm.o
main.o: control.o
main.o: dgmsolver.o
main.o: solver.o
material.o: control.o
mesh.o: control.o
solver.o: control.o
solver.o: material.o
solver.o: angle.o
solver.o: mesh.o
solver.o: state.o
solver.o: sweeper.o
state.o: control.o
state.o: material.o
state.o: mesh.o
state.o: angle.o
sweeper.o: control.o
sweeper.o: material.o
sweeper.o: mesh.o
sweeper.o: angle.o
sweeper.o: state.o
