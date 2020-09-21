#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <tuple>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

/*
Clears screen and moves cursor to home pos on POSIX systems
*/
void clear() {
    std::cout << "\033[2J;" << "\033[1;1H";
}

/*
*/
void printGrid( bool** grid, int* size ) {
    for ( int y = 0; y < size[1]; y++ ) {
        for ( int x = 0; x < size[0]; x++ ) {
            if ( grid[y][x] == true ) {
                std::cout << "0";
            }
            else {
                std::cout << ".";
            }
        }
        std::cout << std::endl;
    }
}

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " [-i input.txt]/[-r] [-o output.txt] [-s 10]\n"
              << "Options:" << std::endl
              << "\t-h, --help\t\tShow this help message and exit" << std::endl
              << "\t-i, --input\t\tProvide an input file for the starting state" << std::endl
              << "\t-r, --random\t\tInstead start with a randomized starting state, provide a seed, 0 will set a random seed" << std::endl
              << "\t-o, --output\t\tOptionally save the final state as a file" << std::endl
              << "\t-s, --steps\t\tThe number of simulation step to take" << std::endl
              << std::endl;
}

int main( int argc, char* argv[] ) {

    int opt;
    char* input;
    char* output;
    bool isRandom = false;
    int seed;
    int steps;
    int size[2] = {10, 10};

    bool** grid;
    bool** d_in;        // The read-only input array for kernel
    bool** d_out;       // The write-only output for kernel

    if ( argc < 2 ) {
        show_usage( argv[0] );
        exit( EXIT_FAILURE );
    }

    while (( opt = getopt(argc, argv, "hi:o:r:s:" )) != -1 ) {
        switch ( opt ) {

        case 'h':
            show_usage( argv[0] );
            exit( EXIT_FAILURE );
            break;

        case 'i':
            input = optarg;
            break;
        
        case 'o':
            output = optarg;
            break;

        case 'r':
            isRandom = true;
            seed = atoi(optarg);
            break;
        
        case 's':
            steps = atoi(optarg);
            break;

        default: /* '?' */
        show_usage( argv[0] );
        exit( EXIT_FAILURE );
        break;

        }
    }

    // Init empty grids on both host and device
    grid = (bool**) malloc( size[1] * sizeof(bool*) );
    cudaMalloc( &d_in, size[1] * size[0] * sizeof(bool*) );
    cudaMalloc( &d_out, size[1] * size[0] * sizeof(bool*) );

    for ( int y = 0; y < size[1]; y++ ) {
        grid[y] = (bool*) malloc( size[0] * sizeof(bool) );

        for ( int x = 0; x < size[0]; x++ ) {
            grid[y][x] = false; // Init host grid to empty
        }
    }

    if ( isRandom ) {
        if ( seed == 0 ) {
            srand( time( NULL ) ); // Set random seed
        }
        else {
            srand( seed );
        }
        
    }

    printGrid( grid, size );

    cuda_hello<<<1,1>>>();
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Clean up memory allocations
    for ( int y = 0; y < size[1]; y++ ) {
        free( grid[y] );
    }
    free(grid);
    cudaFree( d_in );
    cudaFree( d_out );

    exit( EXIT_SUCCESS );
}
