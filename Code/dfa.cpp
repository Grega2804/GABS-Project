#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

// Compute fluctuation F(n) for a single integrated signal and segment size n
double compute_Fn(const double* X, int len, int n) {
    int n_segments = len - n + 1;
    double F_sum = 0.0;

    for (int i = 0; i < n_segments; i++) {
        // Linear regression on segment: Y = a*t + b
        // Using closed-form least squares
        double sum_t = 0.0, sum_x = 0.0, sum_tx = 0.0, sum_t2 = 0.0;
        for (int j = 0; j < n; j++) {
            double t = static_cast<double>(j);
            double x = X[i + j];
            sum_t += t;
            sum_x += x;
            sum_tx += t * x;
            sum_t2 += t * t;
        }
        double nn = static_cast<double>(n);
        double denom = nn * sum_t2 - sum_t * sum_t;
        double a = (nn * sum_tx - sum_t * sum_x) / denom;
        double b = (sum_x - a * sum_t) / nn;

        // Compute sqrt(sum of squared residuals)
        double ss = 0.0;
        for (int j = 0; j < n; j++) {
            double residual = X[i + j] - (a * j + b);
            ss += residual * residual;
        }
        F_sum += std::sqrt(ss);
    }

    return F_sum / n_segments;
}

// Compute F for multiple parcels and segment sizes
// X: (n_parcels, n_samples), segment_sizes: (n_sizes,)
// Returns: (n_sizes,) averaged across parcels
py::array_t<double> compute_F(
    py::array_t<double, py::array::c_style> X,
    py::array_t<int, py::array::c_style> segment_sizes
) {
    auto X_buf = X.unchecked<2>();
    auto sizes_buf = segment_sizes.unchecked<1>();

    int n_parcels = X_buf.shape(0);
    int n_samples = X_buf.shape(1);
    int n_sizes = sizes_buf.shape(0);

    // Integrate each parcel (cumsum)
    std::vector<std::vector<double>> X_int(n_parcels, std::vector<double>(n_samples));
    for (int p = 0; p < n_parcels; p++) {
        X_int[p][0] = X_buf(p, 0);
        for (int t = 1; t < n_samples; t++) {
            X_int[p][t] = X_int[p][t - 1] + X_buf(p, t);
        }
    }

    // Compute F for each segment size, averaged across parcels
    auto result = py::array_t<double>(n_sizes);
    auto res_buf = result.mutable_unchecked<1>();

    for (int s = 0; s < n_sizes; s++) {
        int n = sizes_buf(s);
        double F_avg = 0.0;
        for (int p = 0; p < n_parcels; p++) {
            F_avg += compute_Fn(X_int[p].data(), n_samples, n);
        }
        res_buf(s) = F_avg / n_parcels;
    }

    return result;
}

// Compute DFA alpha: linear fit in log-log space within fitting_range
py::tuple compute_DFA(
    py::array_t<int, py::array::c_style> segment_sizes,
    py::array_t<double, py::array::c_style> F,
    double fit_lo,
    double fit_hi
) {
    auto sizes_buf = segment_sizes.unchecked<1>();
    auto F_buf = F.unchecked<1>();
    int n = sizes_buf.shape(0);

    // Select points within fitting range
    std::vector<double> log_s, log_f;
    for (int i = 0; i < n; i++) {
        double s = static_cast<double>(sizes_buf(i));
        if (s > fit_lo && s < fit_hi) {
            log_s.push_back(std::log10(s));
            log_f.push_back(std::log10(F_buf(i)));
        }
    }

    // Linear regression: log_f = alpha * log_s + intercept
    int m = log_s.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (int i = 0; i < m; i++) {
        sum_x += log_s[i];
        sum_y += log_f[i];
        sum_xy += log_s[i] * log_f[i];
        sum_x2 += log_s[i] * log_s[i];
    }
    double mm = static_cast<double>(m);
    double alpha = (mm * sum_xy - sum_x * sum_y) / (mm * sum_x2 - sum_x * sum_x);
    double intercept = (sum_y - alpha * sum_x) / mm;

    return py::make_tuple(alpha, intercept);
}

PYBIND11_MODULE(dfa_cpp, m) {
    m.doc() = "C++ DFA implementation with pybind11";
    m.def("compute_F", &compute_F,
          "Compute fluctuation function F for multiple parcels and segment sizes",
          py::arg("X"), py::arg("segment_sizes"));
    m.def("compute_DFA", &compute_DFA,
          "Compute DFA exponent alpha",
          py::arg("segment_sizes"), py::arg("F"), py::arg("fit_lo"), py::arg("fit_hi"));
}
