MPI_LSTDFLG = -lstdc++ -llapack -lcblas -lm -lgsl -lgslcblas
MPI_INCLUDE = -I/usr/include/ -I/usr/include/atlas/
MPI_LIB = -L/usr/lib/ -L/usr/lib/atlas/
MPI_OBJS = parallel

all:	${MPI_OBJS}
	rm -f *.o

final.o: final.cpp final.h
	gcc -g -c final.cpp -o final.o ${MPI_INCLUDE}

parallel.o: parallel.cpp final.h
	mpic++ -g -c parallel.cpp -o parallel.o ${MPI_INCLUDE}

parallel: parallel.o final.o
	mpic++ parallel.o final.o -o parallel ${MPI_LIB} ${MPI_LSTDFLG}

clean:
	rm -f *.o
	rm -f ${MPI_OBJS}