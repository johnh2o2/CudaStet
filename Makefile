################################################################################
#
# Build script for project
#
################################################################################
ARCH    	    := 52

# Add source files here
NAME            := custet
################################################################################
# Rules and targets
NVCC=nvcc
CC=g++

CUDA_VERSION=7.5
BLOCK_SIZE=256
VERSION=`cat VERSION.txt`
PYTHON_INCLUDE=-I/usr/include/python2.7


DESTDIR=
PREFIX=/usr/local
SRCDIR=./src
HEADERDIR=./inc
BUILDDIR=./build
LIBDIR=.
BINDIR=.
PYDIR=./cudastet

LIB_NAME=lib$(NAME).so
PY_NAME=$(notdir $(PYDIR))
#GDB_NVCC=-g -G
#GDB_CXX=-g
#DEBUG=-DDEBUG
#SINGLE=-DSINGLE

OPTIMIZE_CPU= -O3
OPTIMIZE_GPU= -Xcompiler -O3 --use_fast_math

DEFS := $(DEBUG) -DBLOCK_SIZE=$(BLOCK_SIZE) -DVERSION=\"$(VERSION)\" $(SINGLE)
NVCCFLAGS := $(GDB_NVCC) $(DEFS) $(OPTIMIZE_GPU) -Xcompiler -fopenmp -Xcompiler -fpic --gpu-architecture=compute_$(ARCH) --gpu-code=sm_$(ARCH),compute_$(ARCH) 
CFLAGS := $(GDB_CXX) $(DEFS) -fPIC -Wall $(OPTIMIZE_CPU)

CUDA_LIBS =`pkg-config --libs cudart-$(CUDA_VERSION)` 

CUDA_INCLUDE =`pkg-config --cflags cudart-$(CUDA_VERSION)`

LIBS := -L$(LIBDIR) $(CUDA_LIBS) -lm
LIBOBJS := dlink.o stetson.o stetson_kernel.o stetson_mean.o utils.o weighting.o
LIBOBJS := $(addprefix $(BUILDDIR)/,$(LIBOBJS))

INSTALL_LIBDIR=$(DESTDIR)$(PREFIX)/lib
INSTALL_INCDIR=$(DESTDIR)$(PREFIX)/include

###############################################################################

CPP_FILES := $(notdir $(wildcard $(SRCDIR)/*.cpp))
CU_FILES  := $(notdir $(wildcard $(SRCDIR)/*.cu))

CPP_OBJ_FILES :=$(CPP_FILES:%.cpp=$(BUILDDIR)/%.o)

CU_OBJ_FILES := $(CU_FILES:%.cu=$(BUILDDIR)/%.o)

INCLUDE := $(CUDA_INCLUDE) -I$(HEADERDIR) 

all : $(NAME) 

$(NAME) : $(CU_OBJ_FILES) $(CPP_OBJ_FILES) $(BUILDDIR)/dlink.o
	$(CC) $(CFLAGS) $(INCLUDE) -o $(BINDIR)/$@ $^ $(LIBS) -largtable2

# This is taken care of in setup.py now
#$(PYDIR)/$(PY_NAME)_wrap.o : 
#	swig -python -c++ -o $(PYDIR)/$(PY_NAME)_wrap.cxx -I$(HEADERDIR) $(PYDIR)/$(PY_NAME).i
#	$(CC) $(CFLAGS) $(INCLUDE) $(PYTHON_INCLUDE) -c $(PYDIR)/$(notdir $(subst .o,.cxx,$@)) -o $(PYDIR)/$(notdir $@)

shlib : $(LIBOBJS)
	$(CC) -shared -o $(LIBDIR)/$(LIB_NAME) $^ $(LIBS)

$(LIB_NAME) : shlib

# This is taken care of in setup.py now
#python : $(PYDIR)/$(PY_NAME)_wrap.o $(LIB_NAME)
#	$(CC) -shared -o $(PYDIR)/_$(PY_NAME).so $^ $(LIBS) -L. -l$(NAME)
#	echo "from $(PYDIR) import *" > $(PYDIR)/__init__.py
	
$(BUILDDIR)/dlink.o : $(CU_OBJ_FILES)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -dlink $^ -o $@

install: shlib
	mkdir -p $(INSTALL_LIBDIR)
	mkdir -p $(INSTALL_INCDIR)/$(NAME)
	cp $(LIB_NAME) $(INSTALL_LIBDIR)/$(LIB_NAME)
	cp $(HEADERDIR)/* $(INSTALL_INCDIR)/$(NAME)

$(CU_OBJ_FILES) : 
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -rdc=true -c $(SRCDIR)/$(notdir $(subst .o,.cu,$@)) -o $(BUILDDIR)/$(notdir $@)

$(CPP_OBJ_FILES) : 
	$(CC) $(CFLAGS) $(INCLUDE) -c $(SRCDIR)/$(notdir $(subst .o,.cpp,$@)) -o $(BUILDDIR)/$(notdir $@)

.PHONY : clean
RM=rm -f

clean-all : clean clean-python
	$(RM) *.dat *.png *.pyc test/*pyc

clean : 
	$(RM) -r $(BUILDDIR)/* $(NAME) $(LIBDIR)/*so 

clean-python :
	$(RM) -r $(PYDIR)/$(PY_NAME).py $(PYDIR)/__init__.py \
		$(PYDIR)/$(PY_NAME)_wrap.cpp $(PYDIR)/*pyc \
                $(PYDIR)/*o dist/ $(PYDIR)/__pycache__

uninstall :
	$(RM) $(INSTALL_LIBDIR)/$(LIB_NAME)
	$(RM) -r $(INSTALL_INCDIR)/$(NAME)

print-%  : ; @echo $* = $($*)
