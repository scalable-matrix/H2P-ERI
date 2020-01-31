CC = icc

USE_LIBXC = 0

BLAS_LIB     = -mkl=parallel
LIBXC_LIBDIR = /home/mkurisu/Workspace/libxc/install/lib
SIMINT_DIR   = /home/mkurisu/Workspace/simint/build-avx/install
YATDFT_DIR   = /home/mkurisu/Workspace/YATDFT
H2PACK_DIR   = /home/mkurisu/Workspace/H2Pack
H2PERI_DIR   = ..

# The order of these lib files matters... 
LIBS     = $(H2PERI_DIR)/src/libH2PERI.a $(H2PACK_DIR)/src/libH2Pack.a $(YATDFT_DIR)/src/libYATDFT.a $(SIMINT_DIR)/lib64/libsimint.a 
INCS     = -I$(H2PERI_DIR)/src/include -I$(H2PACK_DIR)/src/include -I$(YATDFT_DIR)/src -I$(SIMINT_DIR)/include
CFLAGS   = $(INCS) -Wall -g -std=gnu99 -O3 -qopenmp -xHost $(BLAS_LIB)
LDFLAGS  = -g -O3 -xHost -qopenmp $(BLAS_LIB)

ifeq ($(USE_LIBXC), 1)
LDFLAGS += -L$(LIBXC_LIBDIR) -lxc
endif

SRCS = H2ERI-HF.c H2ERI-DFT.c test_SCF_lastD.c test_input_mat.c
OBJS = $(SRCS:.c=.o)
EXES = $(OBJS:.o=.exe)

all: $(EXES)

%.o: %.c
	$(CC) $(CFLAGS)  -c $^ 

%.exe: %.o $(LIBS)
	$(CC) $(LDFLAGS) -o $@ $^ 

clean:
	rm -f $(EXES) $(OBJS)
