#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <fstream>
#include <tuple>
#include <random>
#include <functional>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

__global__ void simulate( bool* input, bool* output, int width, int height, int steps ) {

    int index = threadIdx.x;
    int stride = blockDim.x;

    if ( index >= ( width * height )) {
        // Index out of range
        printf("Out of range: %d\n", index);
        return;
    }

    // Find X and Y
    int y = index / width;
    int x = index % width;
    int size[2] = {height, width};
    //printf("X: %d, Y: %d\n", x, y);

    int count = getNeighbourCount( input, x, y, size );

    if ( input[ index ] ) {
        //printf("alive");
        // Cell is alive
        if ( count == 2 || count == 3 ) {
            // Cell has 2 or 3 neighbours
            output[ index ] = true;
        } else {
            output[ index ] = false;
        }
    } else {
        //printf("dead");
        // Cell is dead
        if ( count == 3 ) {
            // Cell has exactly 3 neighbours
            output[ index ] = true;
        } else {
            output[ index ] = false;
        }
    }
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
              << "\t-p, --play\t\ttOptionally play the simulation in the console" << std::endl
              << std::endl;
}

int main( int argc, char* argv[] ) {

    int opt;
    char* input;
    char* output;
    bool isRandom = false;
    std::ifstream infile;
    std::ofstream outfile;
    bool play = false;
    int seed;
    int steps = 0;
    int size[2] = {20, 10};
    int width, height;
    int gridSize = size[1] * size[0] * sizeof(bool*);

    bool* grid;
    bool* d_in;        // The read-only input array for kernel
    bool* d_out;       // The write-only output for kernel

    if ( argc < 2 ) {
        show_usage( argv[0] );
        exit( EXIT_FAILURE );
    }

    while (( opt = getopt(argc, argv, "hi:o:r:s:p" )) != -1 ) {
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
        
        case 'p':
            play = true;
            break;

        default: /* '?' */
        show_usage( argv[0] );
        exit( EXIT_FAILURE );
        }
    }

    // Init empty grid
    grid = (bool*) malloc( gridSize );
    gpuErrchk( cudaMalloc( &d_in, gridSize ) );
    gpuErrchk( cudaMalloc( &d_out, gridSize ) );

    for ( int y = 0; y < size[1]; y++ ) {
        for ( int x = 0; x < size[0]; x++ ) {
            grid[ y * size[1] + x ] = false; // Init host grid to empty
        }
    }

    if ( isRandom ) {
        if ( ! seed ) {
            seed = std::chrono::steady_clock::now().time_since_epoch().count();
        }
        std::default_random_engine engine(seed);
        std::uniform_int_distribution<> boolGen( 0, 1 );
        for ( int y = 0; y < size[1]; y++ ) {    
            for ( int x = 0; x < size[0]; x++ ) {
                grid[ y * size[1] + x ] = boolGen( engine );
            }
        }
    } else {
        std::ifstream infile("thefile.txt");
    }

    printGrid( grid, size );

    gpuErrchk( cudaMemcpy ( d_in, grid, gridSize, cudaMemcpyHostToDevice ) );

    for (int step = 0; step < steps; step++) {

        simulate<<< 1, 100 >>>( d_in, d_out, size[0], size[1], steps );

        if ( play ) {
            gpuErrchk( cudaMemcpy ( grid, d_out, gridSize, cudaMemcpyDeviceToHost ) );
            gpuErrchk( cudaDeviceSynchronize() );
            clear();
            printGrid( grid, size );
            sleep( 1 );
        }

        gpuErrchk( cudaMemcpy ( d_in, d_out, gridSize, cudaMemcpyHostToHost ) );
    }

    if ( !play ) {
        // Wait for GPU to finish before accessing on host
        gpuErrchk( cudaMemcpy ( grid, d_out, gridSize, cudaMemcpyDeviceToHost ) );
        gpuErrchk( cudaDeviceSynchronize() );
        printGrid( grid, size );
    }

    // Clean up memory allocations
    free( grid );
    cudaFree( d_in );
    cudaFree( d_out );

    exit( EXIT_SUCCESS );
}
