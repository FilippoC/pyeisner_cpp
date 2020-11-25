#pragma once

#include <Python.h>

extern "C"
{
PyObject* py_create_marginals_chart(PyObject* self, PyObject* args);
PyObject* py_delete_marginals_chart(PyObject* self, PyObject* args);
PyObject* py_compute_marginals(PyObject* self, PyObject* args);

PyObject* py_create_argmax_chart(PyObject* self, PyObject* args);
PyObject* py_delete_argmax_chart(PyObject* self, PyObject* args);
PyObject* py_compute_argmax(PyObject* self, PyObject* args);
}