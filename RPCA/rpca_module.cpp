#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <vector>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

// Forward declaration of the solver function
Eigen::MatrixXd solve_rpca(
    const Eigen::MatrixXd& D,
    double lambda,
    double tol
);// Parallel wrapper

std::vector<Eigen::MatrixXd> process_batch(const std::vector<Eigen::MatrixXd>& batch) {
    int num_sets = batch.size();
    std::vector<Eigen::MatrixXd> results(num_sets);

    // Multi-core magic: splits the 800 iterations among CPU cores
    #pragma omp parallel for
    for (int i = 0; i < num_sets; ++i) {
        double lambda = 1.0 / std::sqrt(batch[i].rows());
        results[i] = solve_rpca(batch[i], lambda, 1e-7);
    }
    return results;
}

PYBIND11_MODULE(rpca_module, m) {

    m.def(
        "process_batch",
        &process_batch,
        "RPCA batch processing"
    );
}