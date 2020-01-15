CC = icc

USE_LIBXC = 0

BLAS_LIB       = -mkl=parallel
LIBXC_LIBDIR   = /home/mkurisu/Workspace/libxc/install/lib
YATDFT_INCDIR  = /home/mkurisu/Workspace/YATDFT/src
YATDFT_LIBFILE = /home/mkurisu/Workspace/YATDFT/src/libYATDFT.a
H2PERI_INCDIR  = /home/mkurisu/Workspace/H2P-ERI/src/include
H2PERI_LIBFILE = /home/mkurisu/Workspace/H2P-ERI/src/libH2PERI.a
SIMINT_INCDIR  = /home/mkurisu/Workspace/simint/build-avx/install/include
SIMINT_LIBFILE = /home/mkurisu/Workspace/simint/build-avx/install/lib/libsimint.a

LIBS     = $(YATDFT_LIBFILE) $(H2PERI_LIBFILE) $(SIMINT_LIBFILE)
INCS     = -I$(YATDFT_INCDIR) -I$(H2PERI_INCDIR) -I$(SIMINT_INCDIR)
CFLAGS   = $(INCS) -Wall -g -std=gnu99 -O3 -qopenmp -xHost -mkl $(BLAS_LIB)
LDFLAGS  = -g -O3 -xHost -qopenmp -mkl $(BLAS_LIB)

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
