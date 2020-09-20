rm -r bin/
mkdir bin
nvcc gol.cu -o bin/game-of-life

# Run
./bin/game-of-life