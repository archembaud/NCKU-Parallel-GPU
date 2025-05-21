/*
CSR Demonstration code
Prof. Matthew Smith, NCKU
Using FTCS for steady heat transfer on a 2D Cartesian 4x4 grid, this code
constructs a matrix A and stores its non-zero elements in CSR format.
*/

#include <stdio.h>
#include <stdlib.h>

#define NX 4
#define NY 4
#define N_ROWS (NX*NY)

/*
While we can completely compute how many non-zero elements there are for this
problem, generally we might not know how many non-zero elements there are prior
to mesh, and subsequently, matrix generation. We do know that the maximum number of
non-zero columns in any row for 2D FTCS using 2nd order central differences is 5;
so let's use this initially to set the problem size.
*/
#define N_NON_ZERO (5*NX*NY)

int main(int argc, char *argv[]) {

    int *row_start;
    int *element_column;
    float *element_value;

    /*
       For any given ROW, row_start[ROW+1] - row_start[ROW] tells us the number
       of non-zero columns in that row; row_start[N_ROWS] (the last element) contains
       the total number of non-zero elements in the matrix.
    */  
    row_start = (int*)malloc((N_ROWS+1)*sizeof(int));
    element_column = (int*)malloc(N_NON_ZERO*sizeof(int));
    element_value = (float*)malloc(N_NON_ZERO*sizeof(float));

    // Construct the A matrix
    int element_count = 0;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            /*
            For each cell (i,j), the corresponding column (and row!) in the A matrix is I.
            The neighbours of cell (i,j), together with their associated columns, are shown.
            For instance, cell (i-1,j) refers to column (I-NY).
                              -----------
                              |  I + 1  |
                              | (i,j+1  |
                -------------------------------------
                |  [I - NY]   |    I    |   I + NY   |
                |  (i-1, j)   |  (i,j)  |   (i+1,J)  |
                --------------------------------------
                              |  I - 1  |
                              | (i,j-1) |
                              -----------
            */

            int I = j + i*NX; 
            row_start[I] = element_count;
            /*
            When building up the non-zero columns of row I, the code below
            follows column order. The diagram above shows us this is:
            - (i-1,j) (Column I-NY)
            - (i,j-1) (Column I-1)
            - (i,j)   (Column I, the diagonal element)
            - (i,j+1) (Column I+1)
            - (i+1,j) (Column I+NY)
            Each column only applies if the cell is present; for instance,
            if the cell sits on the boundary, it may not have one (or more)
            of the neighbouring cells.
            */  
            // The first neighbour in order of column location is I-NY (Left)
            if (i > 0) {
                element_value[element_count] = 1.0;
                element_column[element_count] = I-NY;
                element_count++;
            }
            // Next comes I-1 (Underneath I)
            if (j > 0) {
                element_value[element_count] = 1.0;
                element_column[element_count] = I-1;
                element_count++;
            }
            // Then our diagonal, I
            element_value[element_count] = -4.0;
            element_column[element_count] = I;
            element_count++;
            // Then above
            if (j < (NY-1)) {
                element_value[element_count] = 1.0;
                element_column[element_count] = I+1;
                element_count++;                
            }
            // Finally, we have the cell on the right
            if (i < (NX-1)) {
                element_value[element_count] = 1.0;
                element_column[element_count] = I+NY;
                element_count++;                
            }
        }
    }
    row_start[N_ROWS] = element_count;

    // Print the values out for reference
    printf("Values of row_start\n");
    for (int i = 0; i <= N_ROWS; i++) {
        printf("%d\n", row_start[i]);
    }
    printf("Values of element_column, element_value\n");
    for (int i = 0; i < row_start[N_ROWS]; i++) {
        printf("%d, %g\n", element_column[i], element_value[i]);
    }

    printf("Matrix construction complete with %d non-zero elements\n", element_count);
    printf("ROW\tCOLUMN\tVALUE\n");
    printf("--------------------------\n");
    for (int i = 0; i < N_ROWS; i++) {
        int no_elements_in_row = row_start[i+1] - row_start[i];
        for (int j = 0; j < no_elements_in_row; j++) {
            int index = row_start[i] + j;
            printf("%d\t%d\t%f\n", i, element_column[index], element_value[index]);
        }
    }

    // DEMONSTRATION
    // NOTE: There is no reason to construct this entire matrix in the way I've shown below.
    // This is to visually demonstrate how CSR functions.
    float A[NX*NY][NX*NY];
    for (int i = 0; i < (NX*NY); i++) {
        for (int j = 0; j < (NX*NY); j++) {
            A[i][j] = 0.0;
        }
    }

    // Using CSR, add the non-zero elements
    for (int i = 0; i < N_ROWS; i++) {
        int no_elements_in_row = row_start[i+1] - row_start[i];
        for (int j = 0; j < no_elements_in_row; j++) {
            int index = row_start[i] + j;
            A[i][element_column[index]] = element_value[index];
        }
    }

    // Display the results in a matrix view
    for (int i = 0; i < (NX*NY); i++) {
        for (int j = 0; j < (NX*NY); j++) {
            printf("%d", (int)A[i][j]);
            if (j == (NX*NY-1)) {
                printf("\n");
            } else {
                printf(",");
            }
        }
    }

    // Free
    free(row_start);
    free(element_value);
    free(element_column);
}
