LIB   = libH2PERI.a
OBJS  = H2Pack_typedef.o H2Pack_utils.o H2Pack_aux_structs.o \
        H2Pack_ID_compress.o H2Pack_partition.o CMS.o \
        H2ERI_typedef.o H2ERI_shell_operations.o H2ERI_partition.o \
        H2ERI_build.o 

SIMINT_DIR = /home/mkurisu/Workspace/simint/build-avx/install

HOSTCC  = icc
AR      = xiar rcs
DEF     = -DUSE_MKL -mkl
INC     = -I./include -I$(SIMINT_DIR)/include
CFLAGS  = $(INC) -Wall -g -std=gnu99 -O3 -qopenmp -xHost $(DEF)
LDFLAGS = -O3 -xHost -qopenmp 

$(LIB): $(OBJS) 
	$(AR) $@ $^

H2Pack_typedef.o: include/*.h H2Pack_typedef.c
	$(HOSTCC) $(CFLAGS) H2Pack_typedef.c -c 

H2Pack_utils.o: include/*.h H2Pack_utils.c
	$(HOSTCC) $(CFLAGS) H2Pack_utils.c -c 

H2Pack_ID_compress.o: include/*.h H2Pack_ID_compress.c
	$(HOSTCC) $(CFLAGS) H2Pack_ID_compress.c -c 

H2Pack_aux_structs.o: include/*.h H2Pack_aux_structs.c
	$(HOSTCC) $(CFLAGS) H2Pack_aux_structs.c -c 

H2Pack_partition.o: include/*.h H2Pack_partition.c
	$(HOSTCC) $(CFLAGS) H2Pack_partition.c -c 

CMS.o: include/*.h CMS.c
	$(HOSTCC) $(CFLAGS) CMS.c -c 

H2ERI_typedef.o: include/*.h H2ERI_typedef.c
	$(HOSTCC) $(CFLAGS) H2ERI_typedef.c -c 
    
H2ERI_shell_operations.o: include/*.h H2ERI_shell_operations.c
	$(HOSTCC) $(CFLAGS) H2ERI_shell_operations.c -c 

H2ERI_partition.o: include/*.h H2ERI_partition.c
	$(HOSTCC) $(CFLAGS) H2ERI_partition.c -c 

H2ERI_build.o: include/*.h H2ERI_build.c
	$(HOSTCC) $(CFLAGS) H2ERI_build.c -c 

clean:
	rm $(OBJS) $(LIB)