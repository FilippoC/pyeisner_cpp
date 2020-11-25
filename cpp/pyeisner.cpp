#include "pyeisner.h"

#include "marginals.h"
#include "argmax.h"

extern "C"
{
static PyMethodDef python_methods[] =
        {
                {"create_marginals_chart", py_create_marginals_chart, METH_VARARGS, ""},
                {"delete_marginals_chart", py_delete_marginals_chart, METH_VARARGS, ""},
                {"compute_marginals", py_compute_marginals, METH_VARARGS, ""},
                {"create_argmax_chart", py_create_argmax_chart, METH_VARARGS, ""},
                {"delete_argmax_chart", py_delete_argmax_chart, METH_VARARGS, ""},
                {"compute_argmax", py_compute_argmax, METH_VARARGS, ""},
                {NULL, NULL, 0, NULL}
        };

static struct PyModuleDef cModPyDem =
        {
                PyModuleDef_HEAD_INIT,
                "pyeisner",
                NULL,
                -1,
                python_methods
        };
}

PyMODINIT_FUNC
PyInit_pyeisner(void)
{
    return PyModule_Create(&cModPyDem);
}

PyObject* py_create_marginals_chart(PyObject*, PyObject* args)
{
    long size;
    if (!PyArg_ParseTuple(args, "l", &size))
        return NULL;

    diffdp::MarginalsChart* chart = new diffdp::MarginalsChart(size + 1);
    return PyLong_FromVoidPtr((void*) chart);
}

PyObject* py_delete_marginals_chart(PyObject*, PyObject* args)
{
    PyObject* py_chart = nullptr;
    if (!PyArg_ParseTuple(args, "O", &py_chart))
        return NULL;

    diffdp::MarginalsChart* chart = (diffdp::MarginalsChart*) PyLong_AsVoidPtr(py_chart);
    if (chart == nullptr)
        return NULL;
    delete chart;

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* py_compute_marginals(PyObject*, PyObject* args)
{
    long size;
    PyObject* py_multiroot;
    PyObject* py_chart = nullptr;
    PyObject* py_logits = nullptr;
    PyObject* py_marginals = nullptr;
    if (!PyArg_ParseTuple(args, "lOOOO", &size, &py_multiroot, &py_chart, &py_logits, &py_marginals))
        return NULL;

    diffdp::MarginalsChart* chart = (diffdp::MarginalsChart*) PyLong_AsVoidPtr(py_chart);
    float* logits = (float*) PyLong_AsVoidPtr(py_logits);
    float* marginals = (float*) PyLong_AsVoidPtr(py_marginals);

    bool multiroot = PyObject_IsTrue(py_multiroot);
    diffdp::MarginalsAlgorithm alg(chart, !multiroot);
    double log_z = alg.forward(size + 1, [&] (int head, int mod) {
        if (head == 0)
            return logits[(mod - 1) * size + mod - 1];
        else
            return logits[(head - 1) * size + mod - 1];
    });

    for (int head = 0 ; head < size ; ++head)
        for (int mod = 0 ; mod < size ; ++mod)
            if (head == mod)
                marginals[head * size + mod] = alg.output(0, mod + 1);
            else
                marginals[head * size + mod] = alg.output(head + 1, mod + 1);

    return PyFloat_FromDouble(log_z);
}



PyObject* py_create_argmax_chart(PyObject*, PyObject* args)
{
    long size;
    if (!PyArg_ParseTuple(args, "l", &size))
        return NULL;

    ArgmaxChart* chart = new ArgmaxChart(size + 1);
    return PyLong_FromVoidPtr((void*) chart);
}

PyObject* py_delete_argmax_chart(PyObject*, PyObject* args)
{
    PyObject* py_chart = nullptr;
    if (!PyArg_ParseTuple(args, "O", &py_chart))
        return NULL;

    ArgmaxChart* chart = (ArgmaxChart*) PyLong_AsVoidPtr(py_chart);
    if (chart == nullptr)
        return NULL;
    delete chart;

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* py_compute_argmax(PyObject*, PyObject* args)
{
    long size;
    PyObject* py_multiroot;
    PyObject* py_chart = nullptr;
    PyObject* py_logits = nullptr;
    if (!PyArg_ParseTuple(args, "lOOO", &size, &py_multiroot, &py_chart, &py_logits))
        return NULL;

    ArgmaxChart* chart = (ArgmaxChart*) PyLong_AsVoidPtr(py_chart);
    float* logits = (float*) PyLong_AsVoidPtr(py_logits);

    bool multiroot = PyObject_IsTrue(py_multiroot);
    ArgmaxAlgorithm alg(chart, !multiroot);
    auto heads = alg.forward(size + 1, [&] (int head, int mod) {
        if (head == 0)
            return logits[(mod - 1) * size + mod - 1];
        else
            return logits[(head - 1) * size + mod - 1];
    });

    PyObject *list = PyList_New(heads.size());
    for (int i = 0 ; i < heads.size() ; ++i)
    {
        //std::cout << "H:" << heads[i] << "\n" << std::flush;
        PyList_SET_ITEM(list, i, PyLong_FromLong(heads[i]));
    }
    return list;
}
