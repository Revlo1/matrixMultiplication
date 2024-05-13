//Matrix Multiplication using different Memory Management Model
// CMP202
// j.zarrin@abertay.ac.uk
#include <CL/sycl.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
using namespace sycl;


constexpr size_t cols2 = 17; //number of columns in matrix B
constexpr size_t rows1 = 17; //number of rows in matrix A
constexpr size_t cols1 = 18; //number of columns in matrix A
constexpr size_t rows2 = 18; //number of rows in matrix B
constexpr size_t N = 18; // largest num between row and col
constexpr size_t TILE_SIZE = 18; // tile size

constexpr size_t numSG = 16; //size of result?


void tiled_matrix_multiplication(const float A[rows1][cols1], const float B[rows2][cols2], float C[rows1][cols2], queue& q) {
    //buffers
    buffer<float, 2> bufA((float*)A, range<2>(rows1, cols1));
    buffer<float, 2> bufB((float*)B, range<2>(rows2, cols2));
    buffer<float, 2> bufC((float*)C, range<2>(rows1, cols2));
    

    q.submit([&](handler& h) {
        //accessors
        auto accA = bufA.get_access<access::mode::read>(h);
        auto accB = bufB.get_access<access::mode::read>(h);
        auto accC = bufC.get_access<access::mode::write>(h);

        accessor<float, 2, access::mode::read_write, access::target::local> tileA(range<2>(TILE_SIZE, TILE_SIZE), h);
        accessor<float, 2, access::mode::read_write, access::target::local> tileB(range<2>(TILE_SIZE, TILE_SIZE), h);

        h.parallel_for<class TiledMatrixMulKernel>(nd_range<2>(range<2>(N, N), range<2>(TILE_SIZE, TILE_SIZE)), [=](nd_item<2> item) {
            //index variables
            const int globalRow = item.get_global_id(0);
            const int globalCol = item.get_global_id(1);
            const int localRow = item.get_local_id(0);
            const int localCol = item.get_local_id(1);

            float temp = 0.0f;
            
            

            for (int t = 0; t < N; t += TILE_SIZE) {  
                //load tiles into local memory (matrix A)
                if (globalRow < rows1 && localCol < cols1) {
                    tileA[localRow][localCol] = accA[globalRow][t + localCol];
                }
                //load tiles into local memory (matrix B)
                if (localRow < rows2 && globalCol < cols2) {
                    tileB[localRow][localCol] = accB[t + localRow][globalCol];
                }
                item.barrier(access::fence_space::local_space);

                //perform calculation
                for (int k = 0; k < TILE_SIZE; ++k) {
                    temp += tileA[localRow][k] * tileB[k][localCol];
                }
                //sync work items
                item.barrier(access::fence_space::local_space);
            }
            //result matrix C
            accC[globalRow][globalCol] = temp;
            
            });
        });
}

void subgroup_matrix_multiplication(const float A[rows1][cols1], const float B[rows2][cols2], float C[rows1][cols2], queue& q) {
    //buffers
    buffer<float, 2> bufA((float*)A, range<2>(rows1, cols1));
    buffer<float, 2> bufB((float*)B, range<2>(rows2, cols2));
    buffer<float, 2> bufC((float*)C, range<2>(rows1, cols2));
    q.submit([&](handler& h) {
        //accessors
        auto accA = bufA.get_access<access::mode::read>(h);
        auto accB = bufB.get_access<access::mode::read>(h);
        auto accC = bufC.get_access<access::mode::write>(h);
        h.parallel_for(nd_range<2>{range<2>(rows1, cols2), range<2>(numSG, numSG)}, [=](nd_item<2/*rows+cols instead?*/> idx) {
            //index variables
            const int i = idx.get_global_id(0);
            const int j = idx.get_global_id(1);
            float temp = 0.0f;           

            //get subgroup
            auto sg = idx.get_sub_group();

            //perform calculation
            for (int k = 0; k < cols1; ++k) {
                temp += accA[i][k] * accB[k][j];
            }
            //wait for subgroups to finish
            sg.barrier();
            
            //result matrix C
            accC[i][j] = temp;

            sg.barrier();
            });
        });
}


void i_usm_matrix_multiplication(const float* A, const float* B, float* C, queue& q) {
    //uses pointers to matrices
    q.submit([&](handler& h) {
        h.parallel_for<class MatrixMulKernelUSMi>(range<2>(rows1, cols2), [=](id<2> idx) {
            //index variables
            const int i = idx[0];
            const int j = idx[1];
            float temp = 0.0f;
            //perform calculation
            for (int k = 0; k < cols1; ++k) {
                temp += A[i * cols1 + k] * B[k * cols2 + j];
            }
            //result matrix C
            C[i * cols2 + j] = temp;
        });
    });
}


void e_usm_matrix_multiplication(const float* A_host, const float* B_host, float* C_host, queue& q) {
    //allocate memory on the device
    float* A = malloc_device<float>(rows1 * cols1, q);
    float* B = malloc_device<float>(rows2 * cols2, q);
    float* C = malloc_device<float>(rows1 * cols2, q);

    //copy data from host to device
    q.memcpy(A, A_host, sizeof(float) * rows1 * cols1);
    q.memcpy(B, B_host, sizeof(float) * rows2 * cols2);

    //wait for all data to be copied
    q.wait();

    q.submit([&](handler& h) {
        h.parallel_for<class MatrixMulKernelUSMe>(range<2>(rows1, cols2), [=](id<2> idx) {
            //index variable
            const int i = idx[0];
            const int j = idx[1];
            float temp = 0.0f;
            //perform calculation
            for (int k = 0; k < cols1; ++k) {
                temp += A[i * cols1 + k] * B[k * cols2 + j];
            }
            //result matrix C
            C[i * cols2 + j] = temp;
        });       
    });

    //wait for calculations to finish before copying memory
    q.wait();

    //copy the result back to host memory
    q.memcpy(C_host, C, sizeof(float) * rows1 * cols2);

    //wait for memory to be copied before freeing device memory
    q.wait();
    
    //free device memory
    free(A, q);
    free(B, q);
    free(C, q);
}



void matrix_multiplication(const float A[rows1][cols1], const float B[rows2][cols2], float C[rows1][cols2], queue& q) {
    //buffers
    buffer<float, 2> bufA((float*)A, range<2>(rows1, cols1));
    buffer<float, 2> bufB((float*)B, range<2>(rows2, cols2));
    buffer<float, 2> bufC((float*)C, range<2>(rows1, cols2));
    q.submit([&](handler& h) {
        //accessors
        auto accA = bufA.get_access<access::mode::read>(h);
        auto accB = bufB.get_access<access::mode::read>(h);
        auto accC = bufC.get_access<access::mode::write>(h);

        h.parallel_for<class MatrixMulKernel>(range<2>(rows1, cols2), [=](id<2> idx) {
            //index variables
            const int i = idx[0];
            const int j = idx[1];
            float temp = 0.0f;
            //perform calculation
            for (int k = 0; k < cols1; ++k) {
                temp += accA[i][k] * accB[k][j];
            }
            //result matrix C
            accC[i][j] = temp;
            });
        });
}

void clearArray(float* C, queue& q) {
    //uses pointers to matrices
    q.submit([&](handler& h) {
        h.parallel_for<class MatrixMulKernelClear>(range<2>(rows1, cols2), [=](id<2> idx) {
            //index variables
            const int i = idx[0];
            const int j = idx[1];
            //result matrix C
            C[i * cols2 + j] = 0;
            });
        });
}

void test_performance(queue& q) {
    //matrices to be multiplied
    float A[rows1][cols1];
    float B[rows2][cols2];
    float C[rows1][cols2]; //resulting matrix

    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols1; j++) {
            A[i][j] = i * cols1 + j + 1;
        }       
    }

    for (int i = 0; i < rows2; i++) {
        for (int j = 0; j < cols2; j++) {
            B[i][j] = i * cols2 + j + 1 + rows1 * cols1;
        }
    }


    //1
    std::cout << "=-=-=-=-= matrix_multiplication() =-=-=-=-=" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiplication(A, B, C, q);
    q.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "For the Data: " << rows1 << "x" << cols1 << "-Matrix multiplication took " << duration.count() << " nanoseconds.\n";
      
    
    //a
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols1; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //b
    for (int i = 0; i < rows2; ++i) {
        for (int j = 0; j < cols2; ++j) {
            std::cout << B[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //c
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    clearArray(&C[0][0], q);
    std::cout << std::endl;
    std::cout << std::endl;

    //2
    std::cout << "=-=-=-=-= e_usm_matrix_multiplication() =-=-=-=-=" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    e_usm_matrix_multiplication(&A[0][0], &B[0][0], &C[0][0], q);
    q.wait();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "For the Data: " << rows1 << "x" << cols1 << "-Matrix multiplication took " << duration.count() << " nanoseconds.\n";


    //a
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols1; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //b
    for (int i = 0; i < rows2; ++i) {
        for (int j = 0; j < cols2; ++j) {
            std::cout << B[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //c
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    clearArray(&C[0][0], q);
    std::cout << std::endl;
    std::cout << std::endl;

    //3
    std::cout << "=-=-=-=-= i_usm_matrix_multiplication =-=-=-=-=" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    i_usm_matrix_multiplication(&A[0][0], &B[0][0], &C[0][0], q);
    q.wait();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "For the Data: " << rows1 << "x" << cols1 << "-Matrix multiplication took " << duration.count() << " nanoseconds.\n";


    //a
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols1; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //b
    for (int i = 0; i < rows2; ++i) {
        for (int j = 0; j < cols2; ++j) {
            std::cout << B[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //c
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    clearArray(&C[0][0], q);
    std::cout << std::endl;
    std::cout << std::endl;

    
    //4
    std::cout << "=-=-=-=-= tiled_matrix_multiplication =-=-=-=-=" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    tiled_matrix_multiplication(A, B, C, q); //use correct tile size
    q.wait();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "For the Data: " << rows1 << "x" << cols1 << "-Matrix multiplication took " << duration.count() << " nanoseconds.\n";


    //a
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols1; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //b
    for (int i = 0; i < rows2; ++i) {
        for (int j = 0; j < cols2; ++j) {
            std::cout << B[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //c
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    clearArray(&C[0][0], q);
    std::cout << std::endl;
    std::cout << std::endl;
    
    //5
    std::cout << "=-=-=-=-= subgroup_matrix_multiplication =-=-=-=-=" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    subgroup_matrix_multiplication(A, B, C, q);
    q.wait();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "For the Data: " << rows1 << "x" << cols1 << "-Matrix multiplication took " << duration.count() << " nanoseconds.\n";


    //a
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols1; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //b
    for (int i = 0; i < rows2; ++i) {
        for (int j = 0; j < cols2; ++j) {
            std::cout << B[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    //c
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    clearArray(&C[0][0], q);
    std::cout << std::endl;
    std::cout << std::endl;


}

int main() {
    //matrices to be multiplied
    float A[rows1][cols1] = { {1, 2, 3}, {4, 5, 6} };
    float B[rows2][cols2] = { {7, 8}, {9, 10}, {11, 12} };
    float C[rows1][cols2]; //resulting matrix    

    queue q;

    static auto exception_handler = [](sycl::exception_list e_list) {
        for (std::exception_ptr const& e : e_list) {
            try {
                std::rethrow_exception(e);
            }
            catch (std::exception const& e) {
                #if _DEBUG
                std::cout << "Failure" << std::endl;
                #endif
                std::terminate();
            }
        }
        };

    try {
        // queue q(selector, exception_handler);
        queue q(cpu_selector{}, exception_handler);

        // Print out the device information used for the kernel code.
        std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";
        // std::cout << "Vector size: " << a.size() << "\n";

        // Vector addition in SYCL
        // VectorAdd(q, a, b, sum_parallel);
    }
    catch (exception const& e) {
        std::cerr << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
        std::terminate();
    }

    //querying local memory size and maximum work-group size
    auto localMemSize = q.get_device().get_info<info::device::local_mem_size>();
    auto maxWorkGroupSize = q.get_device().get_info<info::device::max_work_group_size>();

    std::cout << "Local Memory Size: " << localMemSize << " bytes\n";
    std::cout << "Max Work-Group Size: " << maxWorkGroupSize << std::endl;
    //float* A = malloc_shared<float>(N * N, q);
    //float* B = malloc_shared<float>(N * N, q);
    //float* C = malloc_shared<float>(N * N, q);
    /*
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i;
            B[i][j] = j;
        }
    }
    */

    //measure performance
    test_performance(q);
    //auto start = std::chrono::high_resolution_clock::now();
    
    //e_usm_matrix_multiplication(&A[0][0], &B[0][0], &C[0][0], q);
    //i_usm_matrix_multiplication(&A[0][0], &B[0][0], &C[0][0], q);
    //tiled_matrix_multiplication(A, B, C, q); //use correct tile size
    //subgroup_matrix_multiplication(A, B, C, q);
    //matrix_multiplication(A, B, C, q);

    //waif for calculation to finish
    q.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    //std::cout << "For the Data: " << rows1 << "x" << cols1 << "-Matrix multiplication took " << duration.count() << " nanoseconds.\n";
    int P = std::min(static_cast<int>(N), 6);
    
    /*

    //matrix A
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols1; j++) {
            std::cout << A[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    //matrix B
    std::cout << "\n" << std::endl;
    for (int i = 0; i < rows2; i++) {
        for (int j = 0; j < cols2; j++) {
            std::cout << B[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    //matrix C (result)
    std::cout << "\n" << std::endl;
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            std::cout << C[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "\n" << std::endl;

    */
    return 0;
}
