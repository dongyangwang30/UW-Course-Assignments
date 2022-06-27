MATRICES_LSTDFLG = -lstdc++ -lm -lgsl -lgslcblas
MATRICES_INCLUDE = -I/usr/include/
MATRICES_LIB = -L/usr/lib/
MATRICES_OBJS = matrices

all:	${MATRICES_OBJS}
	rm -f *.o

matrices.o: matrices.cpp matrices.h
	gcc -g -c matrices.cpp -o matrices.o ${MATRICES_INCLUDE}

main.o: main.cpp matrices.h
	gcc -g -c main.cpp -o main.o

matrices: main.o matrices.o
	gcc main.o matrices.o -o matrices ${MATRICES_LIB} ${MATRICES_LSTDFLG}

clean:
	rm -f *.o
	rm -f ${MATRICES_OBJS}