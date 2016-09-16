################################################################################
#
# Build script for project
#
################################################################################
ARCH    	    := 52

# Add source files here
NAME            := CudaStet
################################################################################
# Rules and targets
NVCC=nvcc
CC=g++

CUDA_VERSION=7.5
REAL_TYPE=double
BLOCK_SIZE=256
VERSION=1.2

SRCDIR=.
HEADERDIR=.
BUILDDIR=.
LIBDIR=.
BINDIR=.

#OPTIMIZE_CPU=
#OPTIMIZE_GPU=
DEBUG=
#GDB_NVCC=-g -G
#GDB_CXX=-g
GDB_NVCC=
GDB_CXX=

#DEBUG=-DDEBUG
OPTIMIZE_CPU= -O3
#OPTIMIZE_CPU=
OPTIMIZE_GPU= -Xcompiler -O3 --use_fast_math
#OPTIMIZE_GPU=
DEFS := $(DEBUG) -DBLOCK_SIZE=$(BLOCK_SIZE) -DVERSION=\"$(VERSION)\" -Dreal_type=$(REAL_TYPE)
NVCCFLAGS := $(GDB_NVCC) $(DEFS) $(OPTIMIZE_GPU) -Xcompiler -fopenmp -Xcompiler -fpic --gpu-architecture=compute_$(ARCH) --gpu-code=sm_$(ARCH),compute_$(ARCH) 
CFLAGS := $(GDB_CXX) $(DEFS) -fPIC -Wall $(OPTIMIZE_CPU)

CUDA_LIBS =`pkg-config --libs cudart-$(CUDA_VERSION)` 

CUDA_INCLUDE =`pkg-config --cflags cudart-$(CUDA_VERSION)` 

LIBS := -L$(LIBDIR) $(CUDA_LIBS) -lm

###############################################################################

CPP_FILES := $(notdir $(wildcard $(SRCDIR)/*.cpp))
CU_FILES  := $(notdir $(wildcard $(SRCDIR)/*.cu))

CPP_OBJ_FILES :=$(CPP_FILES:%.cpp=$(BUILDDIR)/%.o)

CU_OBJ_FILES := $(CU_FILES:%.cu=$(BUILDDIR)/%.o)

INCLUDE := $(CUDA_INCLUDE) -I$(HEADERDIR) 

all : $(NAME) 

$(NAME) : $(CU_OBJ_FILES) $(CPP_OBJ_FILES) dlink.o
	$(CC) $(CFLAGS) $(INCLUDE) -o $(BINDIR)/$@ $^ $(LIBS) -largtable2

dlink.o : $(CU_OBJ_FILES)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -dlink $^ -o $@


$(CU_OBJ_FILES) : 
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -rdc=true -c $(SRCDIR)/$(notdir $(subst .o,.cu,$@)) -o $(BUILDDIR)/$(notdir $@)


$(CPP_OBJ_FILES) : 
	$(CC) $(CFLAGS) $(INCLUDE) -c $(SRCDIR)/$(notdir $(subst .o,.cpp,$@)) -o $(BUILDDIR)/$(notdir $@)

.PHONY : clean
RM=rm -f

clean-all : clean
	$(RM) *.dat *.png
clean : 
	$(RM) *.so *.o $(NAME)

print-%  : ; @echo $* = $($*)
