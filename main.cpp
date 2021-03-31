#include <omp.h>
#include <ctime>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <cstdlib>

#define N 16
#define N1 8

char caracter;
double M1[N1][N1];
double V2, V1[N1];
unsigned seed;
using namespace std;

int numeroAleatorio(int menor, int maior) {
    return rand() % (maior - menor + 1) + menor;
}

void printVector(double a[]) {
// loop through array's rows
    for (int i = 0; i < N1; ++i) {
        cout << setw(9) << a[i];
        cout << endl; // start new line of output
    } // end outer for
} // end function printArray


void printArray(double a[][N1]) {
// loop through array's rows
    for (int i = 0; i < N1; ++i) {
// loop through columns of current row
        for (int j = 0; j < N1; ++j)
            cout << setw(9) << a[i][j];
        cout << endl; // start new line of output
    } // end outer for
} // end function printArray



int main(int argc, char *argv[]) {

    int n = 10000;

    // vetores
    gsl_matrix *a = gsl_matrix_alloc(n, n);
    gsl_vector *x = gsl_vector_alloc(n);
    gsl_vector *y = gsl_vector_alloc(n);

    // gerador randômico
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(rng, time(NULL));

    // inicialização
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            gsl_matrix_set(a, i, j, gsl_rng_uniform(rng));
        }
        gsl_vector_set(x, i, gsl_rng_uniform(rng));
    }

    //gsl_blas_dgemv(CblasNoTrans, 1.0, a, x, 0.0, y);

    // y = A*x
#pragma omp parallel sections
    {
#pragma omp section
        {
            gsl_matrix_const_view as1 = gsl_matrix_const_submatrix(a, 0, 0, n / 2, n);
            gsl_vector_view ys1 = gsl_vector_subvector(y, 0, n / 2);
            gsl_blas_dgemv(CblasNoTrans, 1.0, &as1.matrix, x, 0.0, &ys1.vector);
        }

#pragma omp section
        {
            gsl_matrix_const_view as2 = gsl_matrix_const_submatrix(a, n / 2, 0, (n - n / 2), n);
            gsl_vector_view ys2 = gsl_vector_subvector(y, n / 2, (n - n / 2));
            gsl_blas_dgemv(CblasNoTrans, 1.0, &as2.matrix, x, 0.0, &ys2.vector);
        }
    }

    for (int i = 0; i < n; i++)
        printf("%f\n", gsl_vector_get(y, i));

    gsl_matrix_free(a);
    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_rng_free(rng);

#pragma omp parallel // região paralela
    {
        // id da instância de processamento
        int id = omp_get_thread_num();
        printf("Processo %d, ola !\n", id);
    }
    // E2.1.1.
    // Defina um número de threads maior do que o disponível em sua máquina.
    // Então, rode o código ola.cc e analise a saída. O que você observa?

    printf("\nExercicio E2.1.1, aumentamos para 16 threads");
    printf("\n-----------------------------------------------------\n\n");
    omp_set_num_threads(N); // Define N threads (linha 3)

#pragma omp parallel // região paralela
    {
        int id = omp_get_thread_num(); // id da instância de processamento
        printf("Processo %d, ola !\n", id);
    }

    // E2.1.2.
    // Modifique o código ola.cc de forma que cada thread escreva na tela
    // “Processo ID de NP, olá!”, onde ID é a identificação do thread e NP é o
    // número total de threads disponíveis. O número total de threads pode ser
    // obtido com a função OpenMP omp_get_num_threads();

    printf("\nExercicio E2.1.2, voltamos para 8 threads e identificamos o total de threads");
    printf("\n-----------------------------------------------------\n\n");

    omp_set_num_threads(8); // Define 8 threads
#pragma omp parallel // região paralela
    {
        int id = omp_get_thread_num(); // id da instância de processamento
        int np = omp_get_num_threads();
        printf("Processo %d de %d, ola !\n", id, np);
    }

    // E2.1.3.
    // Faça um código MP para ser executado com 2 threads. O master thread deve ler
    // dois números não nulos a e b, em ponto flutuante. Em paralelo, um dos thread deve
    // computar a-b e o outro deve computar a/b . Por fim, o master thread deve escrever
    // (a-b)+a/b .

    double a1 = 0.0;
    double b = 0.0;

    omp_set_num_threads(2);
    // região paralela
#pragma omp parallel
    {
        int id = omp_get_thread_num(); // id do thread
        if (id == 0) // thread master
        {
            do {
                printf("\n\nDigite o primeiro numero diferente de zero: ");
                scanf("%lf", &a1);
            } while (a1 == 0.0);
        }
    }

#pragma omp parallel
    {
        int id = omp_get_thread_num(); // id do thread
        if (id == 0) // thread master
        {
            do {
                printf("Digite o segundo numero diferente de zero: ");
                scanf("%lf", &b);
            } while (b == 0.0);
        }
    }

#pragma omp parallel
    {
        int id = omp_get_thread_num(); // id do thread
        if (id == 0) // thread master
        {
            printf("Subtracao feito pelo thread master 0 : %f\n", (a1 - b));
        } else if (id == 1) // thread 1
        {
            printf("Divisao feito pelo thread 1 : %f\n", (a1 / b));
        }
    }

#pragma omp parallel
    {
        int id = omp_get_thread_num(); // id do thread
        if (id == 0) // thread master
        {
            printf("O resultado final feito thread master: (a-b)+ a/b = %f\n", (a1 - b) + (a1 / b));
        }
    }

    // E2.1.4. Escreva um código MP para computar a multiplicação de uma matriz com
    // um vetor de elementos. Inicialize todos os elementos com números randômicos em
    // ponto flutuante. Ainda, o código deve ser escrito para um número arbitrário de
    // instâncias de processamento. Por fim, compare o desempenho do código MP com
    // uma versão serial do código.


    //srand((unsigned)time(0)); //para gerar números aleatórios reais.
    srand(10); // seed=10, gera sempre os mesmos números
    int aleatorio = numeroAleatorio(1, 1000);
    //cout << "Numero Aleatorio = " << aleatorio << endl;
    //cout << RAND_MAX;
    cout << "Digite o seed: ";
    cin >> seed;
    srand(seed);


    // loop through array's rows
    for (int i = 0; i < N1; ++i) {
        // loop through columns of current row
        V1[i] = (double) numeroAleatorio(-1000, 1000) + (double) numeroAleatorio(0, 999) / 1000;
        for (int j = 0; j < N1; ++j)
            M1[i][j] = (double) numeroAleatorio(-1000, 1000) + (double) numeroAleatorio(0, 999) / 1000;
    }

    printArray(M1);
    cout << "\n";
    printVector(V1);
    omp_set_num_threads(N1);


    return 0;

}