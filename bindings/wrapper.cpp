#include <pybind11/pybind11.h>

void bind_normal_distribution(pybind11::module_ &);

PYBIND11_MODULE(pyshrew, m) {
    m.doc() = "Python bindings for the shrew library";
    bind_normal_distribution(m);
}
