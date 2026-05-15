#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

using namespace Eigen;

// Soft-thresholding operator (Shrinkage)
double soft_threshold(double x, double epsilon) {
    if (x > epsilon) return x - epsilon;
    if (x < -epsilon) return x + epsilon;
    return 0.0;
}


MatrixXd solve_rpca(const MatrixXd& D, double lambda, double tol = 1e-7) {
    int m = D.rows();
    int n = D.cols();
    
    MatrixXd L = MatrixXd::Zero(m, n);
    MatrixXd S = MatrixXd::Zero(m, n);
    MatrixXd Y = MatrixXd::Zero(m, n);
    
    double mu = 1.25 / D.norm(); 
    double rho = 1.1; 
    
    for (int iter = 0; iter < 1000; ++iter) {
        // 1. Solve for S: S = Shrink(D - L + Y/mu, lambda/mu)
        MatrixXd temp_S = D - L + (Y / mu);
        S = temp_S.unaryExpr([&](double x) { return soft_threshold(x, lambda / mu); });

        // 2. Solve for L: Singular Value Thresholding
        MatrixXd temp_L = D - S + (Y / mu);
        // BDCSVD is Eigen's fastest SVD for these dimensions
        JacobiSVD<MatrixXd> svd(temp_L, ComputeThinU | ComputeThinV);
        VectorXd sigmas = svd.singularValues();
        
        // Shrink singular values
        for (int i = 0; i < sigmas.size(); ++i) {
            sigmas(i) = std::max(0.0, sigmas(i) - 1.0 / mu);
        }
        
        L = svd.matrixU() * sigmas.asDiagonal() * svd.matrixV().transpose();

        // 3. Update dual variable Y and penalty mu
        MatrixXd error = D - L - S;
        Y = Y + mu * error;
        mu *= rho;

        // Check for convergence
        if (error.norm() / D.norm() < tol) break;
    }
    return L; // Return the Low-Rank (clean) 
}