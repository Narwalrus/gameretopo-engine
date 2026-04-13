/*
    batch.cpp -- command line interface to Instant Meshes

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "batch.h"
#include "meshio.h"
#include "dedge.h"
#include "subdivide.h"
#include "meshstats.h"
#include "hierarchy.h"
#include "field.h"
#include "normal.h"
#include "extract.h"
#include "bvh.h"

/* Helper: load a per-vertex float32 map from a binary file */
static bool load_vertex_map(const std::string &path, uint32_t nVerts,
                            float defaultValue, VectorXf &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        cerr << "Error: Could not open vertex map file \"" << path << "\"" << endl;
        return false;
    }
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t nValues = fileSize / sizeof(float);
    if (nValues == 0) {
        cerr << "Error: Vertex map file is empty" << endl;
        fclose(f);
        return false;
    }
    std::vector<float> rawValues(nValues);
    size_t read = fread(rawValues.data(), sizeof(float), nValues, f);
    fclose(f);
    if (read != nValues) {
        cerr << "Error: Could not read vertex map data" << endl;
        return false;
    }
    out.resize(nVerts);
    for (uint32_t i = 0; i < nVerts; ++i) {
        out[i] = (i < nValues) ? rawValues[i] : defaultValue;
    }
    return true;
}

void batch_process(const std::string &input, const std::string &output,
                   int rosy, int posy, Float scale, int face_count,
                   int vertex_count, Float creaseAngle, bool extrinsic,
                   bool align_to_boundaries, int smooth_iter, int knn_points,
                   bool pure_quad, bool deterministic,
                   const std::string &weightMapFile,
                   const std::string &stretchMapFile,
                   const std::string &orientMapFile,
                   bool no_subdivide) {
    cout << endl;
    cout << "Running in batch mode:" << endl;
    cout << "   Input file             = " << input << endl;
    cout << "   Output file            = " << output << endl;
    cout << "   Rotation symmetry type = " << rosy << endl;
    cout << "   Position symmetry type = " << (posy==3?6:posy) << endl;
    cout << "   Crease angle threshold = ";
    if (creaseAngle > 0)
        cout << creaseAngle << endl;
    else
        cout << "disabled" << endl;
    cout << "   Extrinsic mode         = " << (extrinsic ? "enabled" : "disabled") << endl;
    cout << "   Align to boundaries    = " << (align_to_boundaries ? "yes" : "no") << endl;
    cout << "   kNN points             = " << knn_points << " (only applies to point clouds)"<< endl;
    cout << "   Fully deterministic    = " << (deterministic ? "yes" : "no") << endl;
    if (posy == 4)
        cout << "   Output mode            = " << (pure_quad ? "pure quad mesh" : "quad-dominant mesh") << endl;
    cout << endl;

    MatrixXu F;
    MatrixXf V, N;
    VectorXf A;
    std::set<uint32_t> crease_in, crease_out;
    BVH *bvh = nullptr;
    AdjacencyMatrix adj = nullptr;

    /* Load the input mesh */
    load_mesh_or_pointcloud(input, F, V, N);

    bool pointcloud = F.size() == 0;

    Timer<> timer;
    MeshStats stats = compute_mesh_stats(F, V, deterministic);

    if (pointcloud) {
        bvh = new BVH(&F, &V, &N, stats.mAABB);
        bvh->build();
        adj = generate_adjacency_matrix_pointcloud(V, N, bvh, stats, knn_points, deterministic);
        A.resize(V.cols());
        A.setConstant(1.0f);
    }

    if (scale < 0 && vertex_count < 0 && face_count < 0) {
        cout << "No target vertex count/face count/scale argument provided. "
                "Setting to the default of 1/16 * input vertex count." << endl;
        vertex_count = V.cols() / 16;
    }

    if (scale > 0) {
        Float face_area = posy == 4 ? (scale*scale) : (std::sqrt(3.f)/4.f*scale*scale);
        face_count = stats.mSurfaceArea / face_area;
        vertex_count = posy == 4 ? face_count : (face_count / 2);
    } else if (face_count > 0) {
        Float face_area = stats.mSurfaceArea / face_count;
        vertex_count = posy == 4 ? face_count : (face_count / 2);
        scale = posy == 4 ? std::sqrt(face_area) : (2*std::sqrt(face_area * std::sqrt(1.f/3.f)));
    } else if (vertex_count > 0) {
        face_count = posy == 4 ? vertex_count : (vertex_count * 2);
        Float face_area = stats.mSurfaceArea / face_count;
        scale = posy == 4 ? std::sqrt(face_area) : (2*std::sqrt(face_area * std::sqrt(1.f/3.f)));
    }

    cout << "Output mesh goals (approximate)" << endl;
    cout << "   Vertex count           = " << vertex_count << endl;
    cout << "   Face count             = " << face_count << endl;
    cout << "   Edge length            = " << scale << endl;

    MultiResolutionHierarchy mRes;

    if (!pointcloud) {
        /* Subdivide the mesh if necessary */
        VectorXu V2E, E2E;
        VectorXb boundary, nonManifold;
        if (!no_subdivide && (stats.mMaximumEdgeLength*2 > scale || stats.mMaximumEdgeLength > stats.mAverageEdgeLength * 2)) {
            cout << "Input mesh is too coarse for the desired output edge length "
                    "(max input mesh edge length=" << stats.mMaximumEdgeLength
                 << "), subdividing .." << endl;
            build_dedge(F, V, V2E, E2E, boundary, nonManifold);
            subdivide(F, V, V2E, E2E, boundary, nonManifold, std::min(scale/2, (Float) stats.mAverageEdgeLength*2), deterministic);
        } else if (no_subdivide) {
            cout << "Skipping subdivision (--no-subdivide)" << endl;
        }

        /* Compute a directed edge data structure */
        build_dedge(F, V, V2E, E2E, boundary, nonManifold);

        /* Compute adjacency matrix */
        adj = generate_adjacency_matrix_uniform(F, V2E, E2E, nonManifold);

        /* Compute vertex/crease normals */
        if (creaseAngle >= 0)
            generate_crease_normals(F, V, V2E, E2E, boundary, nonManifold, creaseAngle, N, crease_in);
        else
            generate_smooth_normals(F, V, V2E, E2E, nonManifold, N);

        /* Compute dual vertex areas */
        compute_dual_vertex_areas(F, V, V2E, E2E, nonManifold, A);

        mRes.setE2E(std::move(E2E));
    }

    /* Capture vertex count BEFORE moving V into mRes (V is moved-from after) */
    uint32_t capturedVerts = V.cols();

    /* Build multi-resolution hierarrchy */
    mRes.setAdj(std::move(adj));
    mRes.setF(std::move(F));
    mRes.setV(std::move(V));
    mRes.setA(std::move(A));
    mRes.setN(std::move(N));
    mRes.setScale(scale);

    /* Load per-vertex weight map if specified */
    if (!weightMapFile.empty()) {
        uint32_t nVerts = capturedVerts;
        cout << "Loading weight map from \"" << weightMapFile << "\" for " << nVerts << " vertices .. ";
        cout.flush();

        VectorXf vertexScales;
        if (load_vertex_map(weightMapFile, nVerts, 0.5f, vertexScales)) {
            for (uint32_t i = 0; i < nVerts; ++i) {
                float w = vertexScales[i];
                if (w < 0.0f) w = 0.0f;
                if (w > 1.0f) w = 1.0f;
                vertexScales[i] = w;
            }
            mRes.setVertexScale(std::move(vertexScales));
            cout << "done. (" << nVerts << " weights loaded)" << endl;
        } else {
            cout << "FAILED" << endl;
            return;
        }
    }

    /* Load per-vertex stretch map if specified */
    if (!stretchMapFile.empty()) {
        uint32_t nVerts = capturedVerts;
        cout << "Loading stretch map from \"" << stretchMapFile << "\" for " << nVerts << " vertices .. ";
        cout.flush();

        VectorXf vertexStretches;
        if (load_vertex_map(stretchMapFile, nVerts, 1.0f, vertexStretches)) {
            /* Clamp stretch to reasonable range [0.25, 4.0] */
            for (uint32_t i = 0; i < nVerts; ++i) {
                float s = vertexStretches[i];
                if (s < 0.25f) s = 0.25f;
                if (s > 4.0f) s = 4.0f;
                vertexStretches[i] = s;
            }
            mRes.setVertexStretch(std::move(vertexStretches));
            cout << "done. (" << nVerts << " stretches loaded)" << endl;
        } else {
            cout << "FAILED" << endl;
            return;
        }
    }

    mRes.build(deterministic);
    mRes.resetSolution();

    if (align_to_boundaries && !pointcloud) {
        mRes.clearConstraints();
        for (uint32_t i=0; i<3*mRes.F().cols(); ++i) {
            if (mRes.E2E()[i] == INVALID) {
                uint32_t i0 = mRes.F()(i%3, i/3);
                uint32_t i1 = mRes.F()((i+1)%3, i/3);
                Vector3f p0 = mRes.V().col(i0), p1 = mRes.V().col(i1);
                Vector3f edge = p1-p0;
                if (edge.squaredNorm() > 0) {
                    edge.normalize();
                    mRes.CO().col(i0) = p0;
                    mRes.CO().col(i1) = p1;
                    mRes.CQ().col(i0) = mRes.CQ().col(i1) = edge;
                    mRes.CQw()[i0] = mRes.CQw()[i1] = mRes.COw()[i0] =
                        mRes.COw()[i1] = 1.0f;
                }
            }
        }
    }

    /* Load per-vertex orientation hint map (principal curvature directions).
       Fills non-boundary vertices with a soft constraint (weight 0.5) so
       IM's orientation field solver picks the mathematically correct q
       direction instead of an arbitrary one. Critical for cylinders and
       other symmetric surfaces where the orientation field is ambiguous. */
    if (!orientMapFile.empty() && !pointcloud) {
        uint32_t nVerts = capturedVerts;
        cout << "Loading orientation map from \"" << orientMapFile << "\" for "
             << nVerts << " vertices .. ";
        cout.flush();

        /* If boundary constraints weren't set above, we need to clear here */
        if (!align_to_boundaries) {
            mRes.clearConstraints();
        }

        FILE *f = fopen(orientMapFile.c_str(), "rb");
        if (!f) {
            cerr << "Error: Could not open orientation map file" << endl;
            return;
        }
        fseek(f, 0, SEEK_END);
        long fileSize = ftell(f);
        fseek(f, 0, SEEK_SET);

        /* 3 floats per vertex: dx, dy, dz */
        uint32_t nDirs = fileSize / (3 * sizeof(float));
        std::vector<float> rawDirs(nDirs * 3);
        size_t read = fread(rawDirs.data(), sizeof(float), nDirs * 3, f);
        fclose(f);
        if (read != nDirs * 3) {
            cerr << "Error: Could not read orientation map" << endl;
            return;
        }

        /* Seed CQ for interior vertices (boundary vertices already set above) */
        uint32_t seedCount = 0;
        for (uint32_t i = 0; i < nVerts && i < nDirs; ++i) {
            /* Skip if already constrained by boundary (weight > 0) */
            if (mRes.CQw()[i] > 0.5f) continue;

            Vector3f dir(rawDirs[i*3], rawDirs[i*3+1], rawDirs[i*3+2]);
            Float len = dir.norm();
            if (len < 1e-6f) continue;
            dir /= len;

            /* Project onto tangent plane (remove any component along normal) */
            Vector3f n = mRes.N().col(i);
            dir -= n * n.dot(dir);
            len = dir.norm();
            if (len < 1e-6f) continue;
            dir /= len;

            mRes.CQ().col(i) = dir;
            mRes.CQw()[i] = 0.2f;  /* weak soft constraint — hint only */
            seedCount++;
        }
        cout << "done. (seeded " << seedCount << " interior orientations)" << endl;
    }

    /* Propagate constraints if any were set */
    if ((align_to_boundaries || !orientMapFile.empty()) && !pointcloud) {
        mRes.propagateConstraints(rosy, posy);
    }

    if (bvh) {
        bvh->setData(&mRes.F(), &mRes.V(), &mRes.N());
    } else if (smooth_iter > 0) {
        bvh = new BVH(&mRes.F(), &mRes.V(), &mRes.N(), stats.mAABB);
        bvh->build();
    }

    cout << "Preprocessing is done. (total time excluding file I/O: "
         << timeString(timer.reset()) << ")" << endl;

    Optimizer optimizer(mRes, false);
    optimizer.setRoSy(rosy);
    optimizer.setPoSy(posy);
    optimizer.setExtrinsic(extrinsic);

    cout << "Optimizing orientation field .. ";
    cout.flush();
    optimizer.optimizeOrientations(-1);
    optimizer.notify();
    optimizer.wait();
    cout << "done. (took " << timeString(timer.reset()) << ")" << endl;

    std::map<uint32_t, uint32_t> sing;
    compute_orientation_singularities(mRes, sing, extrinsic, rosy);
    cout << "Orientation field has " << sing.size() << " singularities." << endl;
    timer.reset();

    cout << "Optimizing position field .. ";
    cout.flush();
    optimizer.optimizePositions(-1);
    optimizer.notify();
    optimizer.wait();
    cout << "done. (took " << timeString(timer.reset()) << ")" << endl;
    
    //std::map<uint32_t, Vector2i> pos_sing;
    //compute_position_singularities(mRes, sing, pos_sing, extrinsic, rosy, posy);
    //cout << "Position field has " << pos_sing.size() << " singularities." << endl;
    //timer.reset();

    optimizer.shutdown();

    MatrixXf O_extr, N_extr, Nf_extr;
    std::vector<std::vector<TaggedLink>> adj_extr;
    extract_graph(mRes, extrinsic, rosy, posy, adj_extr, O_extr, N_extr,
                  crease_in, crease_out, deterministic);

    MatrixXu F_extr;
    extract_faces(adj_extr, O_extr, N_extr, Nf_extr, F_extr, posy,
            mRes.scale(), crease_out, true, pure_quad, bvh, smooth_iter);
    cout << "Extraction is done. (total time: " << timeString(timer.reset()) << ")" << endl;

    write_mesh(output, F_extr, O_extr, MatrixXf(), Nf_extr);
    if (bvh)
        delete bvh;
}
