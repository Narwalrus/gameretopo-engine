/*
    main_batch.cpp -- Batch-only entry point for Instant Meshes.

    Strips out all GUI dependencies (NanoGUI, GLFW, OpenGL) so the binary
    can be cross-compiled for Windows with MinGW-w64 or built headless
    without a display server.

    Usage: same as the full Instant Meshes binary, but requires -o / --output.
*/

#include "batch.h"
#include "common.h"
#include <tbb/task_scheduler_init.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

int nprocs = -1;

int main(int argc, char **argv) {
    std::vector<std::string> args;
    bool extrinsic = true, dominant = false, align_to_boundaries = false;
    bool help = false, deterministic = false, no_subdivide = false;
    int rosy = 4, posy = 4, face_count = -1, vertex_count = -1;
    uint32_t knn_points = 10, smooth_iter = 2;
    Float crease_angle = -1, scale = -1;
    std::string batchOutput;
    std::string weightMapFile;
    std::string stretchMapFile;
    std::string orientMapFile;

    try {
        for (int i = 1; i < argc; ++i) {
            if (strcmp("--help", argv[i]) == 0 || strcmp("-h", argv[i]) == 0) {
                help = true;
            } else if (strcmp("--deterministic", argv[i]) == 0 || strcmp("-d", argv[i]) == 0) {
                deterministic = true;
            } else if (strcmp("--intrinsic", argv[i]) == 0 || strcmp("-i", argv[i]) == 0) {
                extrinsic = false;
            } else if (strcmp("--boundaries", argv[i]) == 0 || strcmp("-b", argv[i]) == 0) {
                align_to_boundaries = true;
            } else if (strcmp("--threads", argv[i]) == 0 || strcmp("-t", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing thread count!" << std::endl; return -1; }
                nprocs = str_to_uint32_t(argv[i]);
            } else if (strcmp("--smooth", argv[i]) == 0 || strcmp("-S", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing smoothing iterations!" << std::endl; return -1; }
                smooth_iter = str_to_uint32_t(argv[i]);
            } else if (strcmp("--knn", argv[i]) == 0 || strcmp("-k", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing knn argument!" << std::endl; return -1; }
                knn_points = str_to_uint32_t(argv[i]);
            } else if (strcmp("--crease", argv[i]) == 0 || strcmp("-c", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing crease angle!" << std::endl; return -1; }
                crease_angle = str_to_float(argv[i]);
            } else if (strcmp("--rosy", argv[i]) == 0 || strcmp("-r", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing rotation symmetry!" << std::endl; return -1; }
                rosy = str_to_int32_t(argv[i]);
            } else if (strcmp("--posy", argv[i]) == 0 || strcmp("-p", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing position symmetry!" << std::endl; return -1; }
                posy = str_to_int32_t(argv[i]);
                if (posy == 6) posy = 3;
            } else if (strcmp("--scale", argv[i]) == 0 || strcmp("-s", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing scale!" << std::endl; return -1; }
                scale = str_to_float(argv[i]);
            } else if (strcmp("--faces", argv[i]) == 0 || strcmp("-f", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing face count!" << std::endl; return -1; }
                face_count = str_to_int32_t(argv[i]);
            } else if (strcmp("--vertices", argv[i]) == 0 || strcmp("-v", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing vertex count!" << std::endl; return -1; }
                vertex_count = str_to_int32_t(argv[i]);
            } else if (strcmp("--output", argv[i]) == 0 || strcmp("-o", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing output file!" << std::endl; return -1; }
                batchOutput = argv[i];
            } else if (strcmp("--dominant", argv[i]) == 0 || strcmp("-D", argv[i]) == 0) {
                dominant = true;
            } else if (strcmp("--weight-map", argv[i]) == 0 || strcmp("-w", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing weight map file!" << std::endl; return -1; }
                weightMapFile = argv[i];
            } else if (strcmp("--stretch-map", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing stretch map file!" << std::endl; return -1; }
                stretchMapFile = argv[i];
            } else if (strcmp("--orient-map", argv[i]) == 0) {
                if (++i >= argc) { std::cerr << "Missing orient map file!" << std::endl; return -1; }
                orientMapFile = argv[i];
            } else if (strcmp("--no-subdivide", argv[i]) == 0) {
                no_subdivide = true;
            } else {
                if (strncmp(argv[i], "-", 1) == 0) {
                    std::cerr << "Invalid argument: \"" << argv[i] << "\"!" << std::endl;
                    help = true;
                }
                args.push_back(argv[i]);
            }
        }
    } catch (const std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
        help = true;
    }

    if ((posy != 3 && posy != 4) || (rosy != 2 && rosy != 4 && rosy != 6)) {
        std::cerr << "Error: Invalid symmetry type!" << std::endl;
        help = true;
    }

    int nConstraints = 0;
    nConstraints += scale > 0 ? 1 : 0;
    nConstraints += face_count > 0 ? 1 : 0;
    nConstraints += vertex_count > 0 ? 1 : 0;

    if (nConstraints > 1) {
        std::cerr << "Error: Only one of --scale, --face, or --vertices can be used." << std::endl;
        help = true;
    }

    if (batchOutput.empty() && !help) {
        std::cerr << "Error: Batch mode requires --output <file>. This is a batch-only build." << std::endl;
        help = true;
    }

    if (args.size() != 1 || help) {
        std::cout << "Syntax: " << argv[0] << " [options] <input mesh>" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "   -o, --output <file>       Output PLY/OBJ file (REQUIRED)" << std::endl;
        std::cout << "   -t, --threads <count>     Number of threads" << std::endl;
        std::cout << "   -d, --deterministic       Prefer deterministic algorithms" << std::endl;
        std::cout << "   -c, --crease <degrees>    Dihedral angle threshold for creases" << std::endl;
        std::cout << "   -S, --smooth <iter>       Smoothing iterations (default: 2)" << std::endl;
        std::cout << "   -D, --dominant            Tri/quad dominant (not pure)" << std::endl;
        std::cout << "   -i, --intrinsic           Intrinsic mode" << std::endl;
        std::cout << "   -b, --boundaries          Align to boundaries" << std::endl;
        std::cout << "   -r, --rosy <2|4|6>        Orientation symmetry (default: 4)" << std::endl;
        std::cout << "   -p, --posy <4|6>          Position symmetry (default: 4)" << std::endl;
        std::cout << "   -s, --scale <scale>       Desired edge length" << std::endl;
        std::cout << "   -f, --faces <count>       Target face count" << std::endl;
        std::cout << "   -v, --vertices <count>    Target vertex count" << std::endl;
        std::cout << "   -k, --knn <count>         kNN points (point cloud mode)" << std::endl;
        std::cout << "   -w, --weight-map <file>   Per-vertex density weight map (binary float32)" << std::endl;
        std::cout << "       --stretch-map <file>  Per-vertex anisotropic stretch map (binary float32)" << std::endl;
        std::cout << "       --orient-map <file>   Per-vertex orientation hint (binary 3xfloat32: dx,dy,dz)" << std::endl;
        std::cout << "       --no-subdivide        Disable IM's automatic input subdivision" << std::endl;
        std::cout << "   -h, --help                Show this message" << std::endl;
        return help ? 0 : -1;
    }

    tbb::task_scheduler_init init(nprocs == -1 ? tbb::task_scheduler_init::automatic : nprocs);

    try {
        batch_process(args[0], batchOutput, rosy, posy, scale, face_count,
                      vertex_count, crease_angle, extrinsic,
                      align_to_boundaries, smooth_iter, knn_points,
                      !dominant, deterministic, weightMapFile, stretchMapFile,
                      orientMapFile, no_subdivide);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Caught runtime error: " << e.what() << std::endl;
        return -1;
    }
}
