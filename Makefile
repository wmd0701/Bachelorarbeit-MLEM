CC      = gcc
CXX	= g++
NVCC    = nvcc
RM      = rm -f
MPICC	= mpicc
MPICXX	= mpiCC


CFLAGS  = -O3 -g -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ $(BOOST_INC) 
#LFLAGS = -lomp
CUFLAGS =  $(BOOST_INC) -I./helper_files_common -lboost_filesystem -lboost_system -lcublas -lcusparse -lnvidia-ml  -L$(BOOST_LIBDIR)
HELPER_FILES_COMMON = helper_files_common

SOURCES = csr4matrix.cpp scannerconfig.cpp
HEADERS = csr4matrix.hpp vector.hpp matrixelement.hpp scannerconfig.hpp
CUDA_SOURCES = kernel.cu
OBJECTS = $(SOURCES:%.cpp=%.o) 
CUDA_OBJECTS = $(CUDA_SOURCES:%.cu=%.o)

test: main.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(CUFLAGS) $(LFLAGS) $(DEFS) -o $@ main.cu $(OBJECTS) $(CUDA_OBJECTS) 

main.o: main.cu
	$(NVCC) -Xcompiler -fopenmp $(CUFLAGS) $(CFLAGS) $(DEFS) -I. -o $@ -c $<

#kernel.o: kernel.cu
#	$(NVCC) -Xcompiler $(CUFLAGS) $(CFLAGS) $(DEFS) -I. -o $@ -c $<



%.o: $(HELPER_FILES_COMMON)/%.cpp
	$(CXX) $(CFLAGS) $(OMP_FLAGS) $(DEFS) -o $@ -c $<

%.o: $(HELPER_FILES_COMMON)/%.c
	$(CXX) $(CFLAGS) $(OMP_FLAGS) $(DEFS) -o $@ -c $< #-no-legacy-libc
%.o: %.cu
	$(NVCC) -Xcompiler $(OMP_FLAGS) $(CUFLAGS) $(CFLAGS) $(DEFS) -o $@ -c $<



clean:
	- $(RM) -r *.dSYM
	- $(RM) *.o 
