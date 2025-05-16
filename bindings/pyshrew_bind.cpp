#include <pybind11/pybind11.h>

void bind_distributions(pybind11::module_ &);
void bind_random_variable(pybind11::module_ &);
void bind_random_vector(pybind11::module_ &);

PYBIND11_MODULE(pyshrew, m) {
    m.doc() = "Python bindings for the shrew library";
    bind_distributions(m);
    bind_random_variable(m);
    bind_random_vector(m);
}
