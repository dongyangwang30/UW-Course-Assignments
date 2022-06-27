REGRESSIONS_LSTDFLG = -lstdc++ -llapack -lcblas -lm
REGRESSIONS_INCLUDE = -I/usr/include/
REGRESSIONS_LIB = -L/usr/lib/atlas/
REGRESSIONS_OBJS = regressions

all:	${REGRESSIONS_OBJS}
	rm -f *.o

matrices.o: matrices.cpp matrices.h
	gcc -g -c matrices.cpp -o matrices.o ${REGRESSIONS_INCLUDE}

regmodels.o: regmodels.cpp regmodels.h
	gcc -g -c regmodels.cpp -o regmodels.o ${REGRESSIONS_INCLUDE}

main.o: main.cpp matrices.h
	gcc -g -c main.cpp -o main.o

regressions: main.o matrices.o regmodels.o
	gcc main.o regmodels.o matrices.o -o regressions ${REGRESSIONS_LIB} ${REGRESSIONS_LSTDFLG}

clean:
	rm -f *.o
	rm -f ${REGRESSIONS_OBJS}