/* CPython binding for TileCDC. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include "tilecdc.h"

typedef struct {
    PyObject_HEAD
    tilecdc *t;
    tilecdc_params p;
} ChunkerObject;

static void Chunker_dealloc(ChunkerObject *self) {
    if (self->t) tilecdc_free(self->t);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int parse_impl(const char *s, tilecdc_impl *out) {
    if (!s || !strcmp(s, "auto")) *out = TILECDC_IMPL_AUTO;
    else if (!strcmp(s, "amx")) *out = TILECDC_IMPL_AMX;
    else if (!strcmp(s, "avx512")) *out = TILECDC_IMPL_AVX512;
    else if (!strcmp(s, "scalar")) *out = TILECDC_IMPL_SCALAR;
    else return -1;
    return 0;
}

static int Chunker_init(ChunkerObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"min_size", "avg_size", "max_size", "seed", "norm", "impl", NULL};
    tilecdc_params p;
    tilecdc_default_params(&p);
    Py_ssize_t mn = (Py_ssize_t)p.min_size, av = (Py_ssize_t)p.avg_size, mx = (Py_ssize_t)p.max_size;
    PyObject *seed_obj = NULL;
    int norm = p.norm;
    const char *impl = "auto";
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|nnnOis", kwlist, &mn, &av, &mx, &seed_obj, &norm, &impl)) return -1;
    if (mn <= 0 || av <= 0 || mx <= 0) { PyErr_SetString(PyExc_ValueError, "sizes must be positive"); return -1; }
    unsigned long long seed = 0;
    if (seed_obj) {
        seed = PyLong_AsUnsignedLongLong(seed_obj);
        if (PyErr_Occurred()) { PyErr_Clear(); PyErr_SetString(PyExc_ValueError, "seed must be an integer in [0, 2**64)"); return -1; }
    }
    p.min_size = (size_t)mn; p.avg_size = (size_t)av; p.max_size = (size_t)mx;
    p.seed = seed; p.norm = norm;
    if (parse_impl(impl, &p.impl)) { PyErr_SetString(PyExc_ValueError, "impl must be auto, amx, avx512 or scalar"); return -1; }
    tilecdc *t = tilecdc_new(&p);   /* build the new chunker before touching the old one */
    if (!t) {
        if (p.impl != TILECDC_IMPL_AUTO && !tilecdc_impl_available(p.impl))
            PyErr_Format(PyExc_RuntimeError, "implementation '%s' is not available on this CPU", impl);
        else
            PyErr_Format(PyExc_ValueError, "invalid parameters: need 64 <= min_size < avg_size <= max_size, avg_size <= %d, 0 <= norm <= 4", TILECDC_MAX_AVG);
        return -1;
    }
    if (self->t) tilecdc_free(self->t);
    self->t = t;
    self->p = p;
    return 0;
}

static int ready(ChunkerObject *self) {
    if (self->t) return 1;
    PyErr_SetString(PyExc_RuntimeError, "Chunker is not initialised");
    return 0;
}

static PyObject *Chunker_cut(ChunkerObject *self, PyObject *arg) {
    if (!ready(self)) return NULL;
    Py_buffer b;
    if (PyObject_GetBuffer(arg, &b, PyBUF_SIMPLE) < 0) return NULL;
    size_t r;
    Py_BEGIN_ALLOW_THREADS
    r = tilecdc_cut(self->t, (const uint8_t *)b.buf, (size_t)b.len);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&b);
    return PyLong_FromSize_t(r);
}

/* chunk(data) -> list of chunk lengths */
static PyObject *Chunker_chunk(ChunkerObject *self, PyObject *arg) {
    if (!ready(self)) return NULL;
    Py_buffer b;
    if (PyObject_GetBuffer(arg, &b, PyBUF_SIMPLE) < 0) return NULL;
    size_t cap = (size_t)b.len / self->p.min_size + 2;
    size_t *ends = PyMem_RawMalloc(cap * sizeof(size_t));
    if (!ends) { PyBuffer_Release(&b); return PyErr_NoMemory(); }
    size_t n;
    Py_BEGIN_ALLOW_THREADS
    n = tilecdc_chunk_all(self->t, (const uint8_t *)b.buf, (size_t)b.len, ends, cap);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&b);
    PyObject *out = PyList_New((Py_ssize_t)n);
    if (!out) { PyMem_RawFree(ends); return NULL; }
    size_t s = 0;
    for (size_t i = 0; i < n; i++) {
        PyObject *v = PyLong_FromSize_t(ends[i] - s);
        if (!v) { Py_DECREF(out); PyMem_RawFree(ends); return NULL; }
        PyList_SET_ITEM(out, (Py_ssize_t)i, v);
        s = ends[i];
    }
    PyMem_RawFree(ends);
    return out;
}

static PyObject *Chunker_get_impl(ChunkerObject *self, void *c) { (void)c; if (!ready(self)) return NULL; return PyUnicode_FromString(tilecdc_impl_name(tilecdc_get_impl(self->t))); }
static PyObject *Chunker_get_min(ChunkerObject *self, void *c) { (void)c; return PyLong_FromSize_t(self->p.min_size); }
static PyObject *Chunker_get_avg(ChunkerObject *self, void *c) { (void)c; return PyLong_FromSize_t(self->p.avg_size); }
static PyObject *Chunker_get_max(ChunkerObject *self, void *c) { (void)c; return PyLong_FromSize_t(self->p.max_size); }

static PyGetSetDef Chunker_getset[] = {
    {"impl", (getter)Chunker_get_impl, NULL, "implementation in use: amx, avx512 or scalar", NULL},
    {"min_size", (getter)Chunker_get_min, NULL, NULL, NULL},
    {"avg_size", (getter)Chunker_get_avg, NULL, NULL, NULL},
    {"max_size", (getter)Chunker_get_max, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyMethodDef Chunker_methods[] = {
    {"cut", (PyCFunction)Chunker_cut, METH_O, "cut(data) -> length of the first chunk (pass >= max_size bytes unless at end of stream)"},
    {"chunk", (PyCFunction)Chunker_chunk, METH_O, "chunk(data) -> list of chunk lengths covering all of data"},
    {NULL, NULL, 0, NULL},
};

static PyTypeObject ChunkerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tilecdc._tilecdc.Chunker",
    .tp_basicsize = sizeof(ChunkerObject),
    .tp_dealloc = (destructor)Chunker_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Chunker(min_size=2048, avg_size=8192, max_size=65536, seed=0, norm=1, impl='auto')",
    .tp_methods = Chunker_methods,
    .tp_getset = Chunker_getset,
    .tp_init = (initproc)Chunker_init,
    .tp_new = PyType_GenericNew,
};

static PyObject *py_available(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    PyObject *l = PyList_New(0);
    tilecdc_impl all[3] = {TILECDC_IMPL_AMX, TILECDC_IMPL_AVX512, TILECDC_IMPL_SCALAR};
    for (int i = 0; i < 3; i++)
        if (tilecdc_impl_available(all[i])) {
            PyObject *s = PyUnicode_FromString(tilecdc_impl_name(all[i]));
            PyList_Append(l, s);
            Py_DECREF(s);
        }
    return l;
}

static PyMethodDef module_methods[] = {
    {"available_impls", py_available, METH_NOARGS, "implementations usable on this CPU, fastest first"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moddef = {PyModuleDef_HEAD_INIT, "_tilecdc", "TileCDC native core", -1, module_methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__tilecdc(void) {
    if (PyType_Ready(&ChunkerType) < 0) return NULL;
    PyObject *m = PyModule_Create(&moddef);
    if (!m) return NULL;
    Py_INCREF(&ChunkerType);
    PyModule_AddObject(m, "Chunker", (PyObject *)&ChunkerType);
    PyModule_AddStringConstant(m, "__version__", TILECDC_VERSION);
    PyModule_AddIntConstant(m, "MAX_AVG_SIZE", TILECDC_MAX_AVG);
    return m;
}
