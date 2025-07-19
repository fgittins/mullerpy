from typing import cast
from unittest import TestCase

import numpy

from mullerpy import muller


def f(x: complex, n: complex, p: complex) -> complex:
    return x**n - p


def g(x: complex) -> complex:
    return f(x, 2, 612)


def h(x: complex) -> complex:
    return f(x, 3, 1)


def i(x: complex) -> complex:
    return cast("complex", numpy.exp(-x) * numpy.sin(x))


class TestMuller(TestCase):
    def test_quadratic_roots(self) -> None:
        x = (10, 20, 30)
        res = muller(g, x)
        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        x = (-10, -20, -30)
        res = muller(g, x)
        self.assertAlmostEqual(res.root, -(612 ** (1 / 2)), delta=1e-5)

    def test_sine_roots(self) -> None:
        x = (1, 2, 3)
        res = muller(numpy.sin, x)
        self.assertAlmostEqual(res.root, numpy.pi, delta=1e-5)

        x = (2, 4, 6)
        res = muller(numpy.sin, x)
        self.assertAlmostEqual(res.root, 2 * numpy.pi, delta=1e-5)

    def test_exp_sine_roots(self) -> None:
        x = (-2, -3, -4)
        res = muller(i, x)
        self.assertAlmostEqual(res.root, -numpy.pi, delta=1e-5)

        y = (-1, 0, 1 / 2)
        res = muller(i, y)
        self.assertAlmostEqual(res.root, 0, delta=1e-5)

        z = (-1, 0, 1)
        res = muller(i, z)
        self.assertAlmostEqual(res.root, numpy.pi, delta=1e-5)

    def test_args(self) -> None:
        n, p = 2, 612

        x = (10, 20, 30)
        res = muller(f, x, args=(n, p))
        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        x = (-10, -20, -30)
        res = muller(f, x, args=(n, p))
        self.assertAlmostEqual(res.root, -(612 ** (1 / 2)), delta=1e-5)

    def test_complex_roots(self) -> None:
        x = (-1, (-1 + 1j) / 2, 1j)
        res = muller(h, x)
        self.assertAlmostEqual(
            res.root, (-1 + 3 ** (1 / 2) * 1j) / 2, delta=1e-5
        )

        x = (-1, -(1 + 1j) / 2, -1j)
        res = muller(h, x)
        self.assertAlmostEqual(
            res.root, -(1 + 3 ** (1 / 2) * 1j) / 2, delta=1e-5
        )

    def test_tol(self) -> None:
        x = (10, 20, 30)

        res = muller(g, x, xtol=0)
        self.assertTrue(res.is_converged)
        self.assertIn("function", res.flag)

        res = muller(g, x, ftol=0)
        self.assertTrue(res.is_converged)
        self.assertIn("root", res.flag)

    def test_errors(self) -> None:
        x = (10, 20)

        with self.assertRaises(ValueError):
            muller(g, x)

        y = (10, 20, 30)

        with self.assertRaises(ValueError):
            muller(g, y, xtol=-1e-5)

        with self.assertRaises(ValueError):
            muller(g, y, ftol=-1e-5)

        with self.assertRaises(ValueError):
            muller(g, y, maxiter=0)

    def test_iterable(self) -> None:
        n, p = 2, 612

        v = [10, 20, 30]
        res = muller(g, v)
        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        w = {10, 20, 30}
        res = muller(g, w)
        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        x = range(10, 31, 10)
        res = muller(g, x)
        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        y = numpy.array([10, 20, 30], dtype=numpy.int64)
        res = muller(g, y)
        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        z = (10, 20, 30)

        res = muller(f, z, args=[n, p])
        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        res = muller(f, z, args={n, p})
        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        res = muller(f, z, args=numpy.array([n, p], dtype=numpy.int64))
        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)
