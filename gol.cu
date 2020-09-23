#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <tuple>
#include <random>
#include <functional>

__device__ int getNeighbourCount( bool** input, int x, int y, int* size ) {
    int count = 0;

    if ( ( x - 1 ) < 0 ) {
        if ( input[y][ size[1] ] ) { count++; }
    } else {
        if ( input[y][x - 1] ) { count++; }
    }
    if ( ( x + 1 ) >= size[1] ) {
        if ( input[y][0] ) { count++; }
    } else {
        if ( input[y][x + 1] ) { count++; }
    }
    if ( ( y - 1 ) < 0 ) {
        if ( input[ size[0] ][x] ) { count++; }
    } else {
        if ( input[y - 1][x] ) { count++; }
    }
    if ( ( x + 1 ) >= size[1] ) {
        if ( input[0][x] ) { count++; }
    } else {
        if ( input[y + 1][x] ) { count++; }
    }
    return count;
}

__global__ void simulate( bool* input, bool** output, int* size, int steps ) {

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
        }
    }

    // Init empty grid
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
        auto gen = std::bind(   std::uniform_int_distribution<>( 0,1 ),
                                std::default_random_engine() );
        for ( int y = 0; y < size[1]; y++ ) {    
            for ( int x = 0; x < size[0]; x++ ) {
                grid[y][x] = gen();
            }
        }
    }

    printGrid( grid, size );

    simulate<<<1,1>>>();
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Clean up memory allocations
    for ( int y = 0; y < size[1]; y++ ) {
        free( grid[y] );
    }
    free( grid );
    cudaFree( d_in );
    cudaFree( d_out );

    exit( EXIT_SUCCESS );
}
