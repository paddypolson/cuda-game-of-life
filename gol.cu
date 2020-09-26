#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <tuple>
#include <random>
#include <functional>

__device__ bool getCell( bool* input, int x, int y, int* size ) {
    if ( x < 0 ) { x = x + size[1]; }
    else if ( x >= size[1] ) { x = x - size[1]; }
    if ( y < 0 ) { y = y + size[0]; }
    else if ( y >= size[0] ) { y = y - size[0]; }
    return input[ y * size[0] + x ];
}

__device__ int getNeighbourCount( bool* input, int x, int y, int* size ) {
    int count = 0;
    if ( getCell( input, x - 1, y, size )) { count++; }
    if ( getCell( input, x + 1, y, size )) { count++; }
    if ( getCell( input, x, y - 1, size )) { count++; }
    if ( getCell( input, x, y + 1, size )) { count++; }
    return count;
}

__global__ void simulate( bool* input, bool* output, int* size, int steps ) {

    int index = threadIdx.x;
    int stride = blockDim.x;

}

/*
Clears screen and moves cursor to home pos on POSIX systems
*/
void clear() {
    std::cout << "\033[2J;" << "\033[1;1H";
}

/*
*/
void printGrid( bool* grid, int* size ) {
    for ( int y = 0; y < size[1]; y++ ) {
        for ( int x = 0; x < size[0]; x++ ) {
            if ( grid[ y * size[1] + x ] == true ) {
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
    int steps = 0;
    int size[2] = {10, 10};

    bool* grid;
    bool* d_in;        // The read-only input array for kernel
    bool* d_out;       // The write-only output for kernel

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
        }
    }

    // Init empty grid
    grid = (bool*) malloc( size[1] * size[0] * sizeof(bool*) );
    cudaMalloc( &d_in, size[1] * size[0] * sizeof(bool*) );
    cudaMalloc( &d_out, size[1] * size[0] * sizeof(bool*) );

    for ( int y = 0; y < size[1]; y++ ) {
        for ( int x = 0; x < size[0]; x++ ) {
            grid[ y * size[1] + x ] = false; // Init host grid to empty
        }
    }

    if ( isRandom ) {
        auto gen = std::bind(   std::uniform_int_distribution<>( 0,1 ),
                                std::default_random_engine() );
        for ( int y = 0; y < size[1]; y++ ) {    
            for ( int x = 0; x < size[0]; x++ ) {
                grid[ y * size[1] + x ] = gen();
            }
        }
    }

    printGrid( grid, size );

    simulate<<< 1, 1 >>>( d_in, d_out, size, steps );
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Clean up memory allocations
    free( grid );
    cudaFree( d_in );
    cudaFree( d_out );

    exit( EXIT_SUCCESS );
}
