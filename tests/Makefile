CC      = icc

USE_MKL      = 1
USE_OPENBLAS = 0
USE_LIBXC    = 0

LIBXC_LIBDIR         = /home/mkurisu/Workspace/libxc/install/lib
H2PERI_INSTALL_DIR   = /home/mkurisu/Workspace/H2P-ERI
SIMINT_INSTALL_DIR   = /home/mkurisu/Workspace/simint/build-avx/install
YATDFT_INSTALL_DIR   = /home/mkurisu/Workspace/YATDFT
H2PACK_INSTALL_DIR   = /home/mkurisu/Workspace/H2Pack
OPENBLAS_INSTALL_DIR = /home/mkurisu/Workspace/OpenBLAS-0.3.8/install

# The order of these lib files matters... 
DEF      = -DUSE_AVX
LIBS     = $(H2PERI_INSTALL_DIR)/src/libH2PERI.a $(H2PACK_INSTALL_DIR)/src/libH2Pack.a $(YATDFT_INSTALL_DIR)/src/libYATDFT.a $(SIMINT_INSTALL_DIR)/lib64/libsimint.a 
INCS     = -I$(H2PERI_INSTALL_DIR)/src/include -I$(H2PACK_INSTALL_DIR)/src/include -I$(YATDFT_INSTALL_DIR)/src/include -I$(SIMINT_INSTALL_DIR)/include
CFLAGS   = $(INCS) -Wall -g -std=gnu99 -O3 $(DEF)
LDFLAGS  = -g -O3

ifeq ($(CC), icc)
CFLAGS  += -qopenmp -xHost
endif

ifeq ($(CC), gcc)
CFLAGS  += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
endif

ifeq ($(USE_MKL), 1)
DEF     += -DUSE_MKL
CFLAGS  += -mkl
LDFLAGS += -mkl -qopenmp
endif

ifeq ($(USE_OPENBLAS), 1)
DEF     += -DUSE_OPENBLAS
LIBS    += $(OPENBLAS_INSTALL_DIR)/lib/libopenblas.a
INCS    += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -fopenmp -lm
endif

ifeq ($(USE_LIBXC), 1)
LDFLAGS += -L$(LIBXC_LIBDIR) -lxc
endif

SRCS = H2ERI-HF.c H2ERI-DFT.c test_SCF_lastD.c test_input_mat.c test_perf.c
OBJS = $(SRCS:.c=.o)
EXES = $(OBJS:.o=.exe)

all: $(EXES)

%.o: %.c
	$(CC) $(CFLAGS)  -c $^ 

%.exe: %.o $(LIBS)
	$(CC) -o $@ $^ $(LDFLAGS) 

clean:
	rm -f $(EXES) $(OBJS)
