import numpy as np
import torch


def test_easy():
    def easy(x, y):
        aaa = torch.add(x, y)
        return aaa

    traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024)))

    a = torch.rand(1024)
    b = torch.rand(1024)
    x = traced(a, b)
    np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())


def test_three_arg():
    def easy(x, y, z):
        aaa = torch.add(x, y)
        bbb = torch.add(aaa, z)
        return bbb

    traced = torch.jit.trace(
        easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
    )

    a = torch.rand(1024)
    b = torch.rand(1024)
    c = torch.rand(1024)
    x = traced(a, b, c)
    npr = a.numpy() + b.numpy() + c.numpy()
    np.testing.assert_allclose(npr, x.numpy())


def test_all_combos():
    def easy(x, y, z):
        a = torch.add(x, y)
        b = torch.add(a, z)
        c = torch.add(x, b)
        d = torch.add(c, a)
        return d

    def np_easy(x, y, z):
        a = x + y
        b = a + z
        c = x + b
        d = c + a
        return d

    traced = torch.jit.trace(
        easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
    )

    a = torch.rand(1024)
    b = torch.rand(1024)
    c = torch.rand(1024)
    x = traced(a, b, c)
    npr = np_easy(a.numpy(), b.numpy(), c.numpy())
    np.testing.assert_allclose(npr, x.numpy())


def test_rank_two():
    def easy(x, y, z):
        a = torch.add(x, y)
        b = torch.add(a, z)
        c = torch.add(x, b)
        d = torch.add(c, a)
        return d

    def np_easy(x, y, z):
        a = x + y
        b = a + z
        c = x + b
        d = c + a
        return d

    shape = 32, 32
    traced = torch.jit.trace(
        easy, (torch.rand(shape), torch.rand(shape), torch.rand(shape))
    )

    a = torch.rand(shape)
    b = torch.rand(shape)
    c = torch.rand(shape)
    x = traced(a, b, c)
    npr = np_easy(a.numpy(), b.numpy(), c.numpy())
    np.testing.assert_allclose(npr, x.numpy())


def test_broadcast():
    def easy(x, y, z):
        a = torch.add(x, y)
        b = torch.add(a, z)
        return b

    def np_easy(x, y, z):
        a = x + y
        b = a + z
        return b

    N = 32
    traced = torch.jit.trace(easy, (torch.rand(N, N), torch.rand(N), torch.rand(N, N)))

    a = torch.rand(N, N)
    b = torch.rand(N)
    c = torch.rand(N, N)
    x = traced(a, b, c)
    npr = np_easy(a.numpy(), b.numpy(), c.numpy())
    np.testing.assert_allclose(npr, x.numpy())


def test_broadcast_2():
    zero = torch.tensor([0.0], dtype=torch.float)

    def foo(x, y, z):
        aaa = torch.add(x, y)
        bbb = torch.add(zero, aaa)
        return torch.add(bbb, z)

    def foo_np(x, y, z):
        a = x + y
        b = zero.numpy() + a
        return b + z

    x = torch.rand(3, 4)
    y = torch.ones(3, 1)
    z = torch.rand(4)
    traced = torch.jit.trace(foo, (x, y, z))

    r = traced(x, y, z)
    rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
    np.testing.assert_allclose(r, rnp)


def test_broadcast_big2():
    zero = torch.tensor([0.0], dtype=torch.float)

    def foo(x, y, z):
        aaa = torch.add(x, y)
        bbb = torch.add(zero, aaa)
        return torch.add(bbb, z)

    def foo_np(x, y, z):
        a = x + y
        b = zero.numpy() + a
        return b + z

    x = torch.rand(32, 1024)
    y = torch.ones(32, 1)
    z = torch.rand(1024)
    traced = torch.jit.trace(foo, (x, y, z))

    r = traced(x, y, z)
    rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
    np.testing.assert_allclose(r, rnp)


def test_alpha():
    def alpha(x):
        aaa = torch.add(x, x, alpha=2.0)
        return aaa

    traced = torch.jit.trace(alpha, (torch.tensor([1.0])))

    a = torch.tensor([1.0])
    x = traced(a)
    np.testing.assert_allclose(a.numpy() + 2.0 * a.numpy(), x.numpy())


def test_constant():
    def constant(x):
        bbb = torch.tensor([1.0])
        aaa = torch.add(x, bbb)
        return aaa

    traced = torch.jit.trace(constant, (torch.tensor([1.0])))

    a = torch.tensor([1.0])
    x = traced(a)
    np.testing.assert_allclose(a.numpy() + 1.0, x.numpy())


def test_add_sub():
    def easy(x, y, z):
        aaa = torch.add(x, y)
        bbb = torch.sub(aaa, z)
        return bbb

    traced = torch.jit.trace(
        easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
    )

    a = torch.rand(1024)
    b = torch.rand(1024)
    c = torch.rand(1024)
    x = traced(a, b, c)
    np.testing.assert_allclose(a.numpy() + b.numpy() - c.numpy(), x.numpy())


def test_promotion():
    def easy(x, y):
        aaa = torch.add(x, y)
        return aaa

    traced = torch.jit.trace(
        easy,
        (torch.zeros(1024, dtype=torch.int32), torch.rand(1024, dtype=torch.float32)),
    )

    a = torch.zeros(1024, dtype=torch.int32)
    b = torch.rand(1024, dtype=torch.float32)
    x = traced(a, b)
    np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())


def test_eq():
    def easy(x, y):
        c = torch.eq(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    a = torch.zeros(1024, dtype=torch.int32)
    b = torch.zeros(1024, dtype=torch.int32)
    x = traced(a, b)
    np.testing.assert_allclose(np.ones(1024), x.numpy())


def test_ne():
    def easy(x, y):
        c = torch.ne(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    a = torch.zeros(1024, dtype=torch.int32)
    b = torch.ones(1024, dtype=torch.int32)
    x = traced(a, b)
    np.testing.assert_allclose(np.ones(1024), x.numpy())


def test_ge():
    def easy(x, y):
        c = torch.ge(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    aa = np.array(1024, dtype=int)
    aa.fill(5)
    a = torch.from_numpy(aa)
    b = torch.zeros(1024, dtype=torch.int32)
    x = traced(a, b)
    np.testing.assert_allclose(np.ones(1024), x.numpy())


def test_gt():
    def easy(x, y):
        c = torch.gt(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    a = torch.ones(1024, dtype=torch.int32)
    b = torch.zeros(1024, dtype=torch.int32)
    x = traced(a, b)
    np.testing.assert_allclose(np.ones(1024), x.numpy())


def test_le():
    def easy(x, y):
        c = torch.le(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    aa = np.array(1024, dtype=int)
    aa.fill(5)
    a = torch.from_numpy(aa)
    b = torch.zeros(1024, dtype=torch.int32)
    x = traced(a, b)
    np.testing.assert_allclose(np.zeros(1024), x.numpy())


def test_lt():
    def easy(x, y):
        c = torch.lt(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    a = torch.ones(1024, dtype=torch.int32)
    b = torch.zeros(1024, dtype=torch.int32)
    x = traced(a, b)
    np.testing.assert_allclose(np.zeros(1024), x.numpy())


def test_min_max():
    def test(x, y):
        return torch.max(torch.min(x, y), torch.tensor([4.0]))

    traced = torch.jit.trace(test, (torch.zeros(1024), torch.zeros(1024)))
    a = 8.0 * torch.rand(1024)
    b = 8.0 * torch.rand(1024)
    np.testing.assert_allclose(
        traced(a, b),
        np.maximum(np.minimum(a.numpy(), b.numpy()), [4.0]))


def test_clamp():
    def test(x):
        return torch.clamp(x + 3.0, 0.0, 6.0)

    traced = torch.jit.trace(test, (torch.zeros(1024)))
    a = 20.0 * torch.rand(1024) - 10.0
    an = a.numpy()
    np.testing.assert_allclose(
        traced(a),
        np.clip(an + 3.0, 0.0, 6.0))


def test_reps():
    def easy(x, y):
        c = torch.add(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024)))

    for _ in range(32):
        a = torch.ones(1024)
        b = torch.zeros(1024)
        x = traced(a, b)
        np.testing.assert_allclose(np.ones(1024), x.numpy())


def test_add_const_rhs():
    def test(x):
        return x + 3.0

    traced = torch.jit.trace(test, torch.rand(4))
    x = torch.rand(4)
    y = traced(x)
    np.testing.assert_allclose(x.numpy() + 3.0, y.numpy())


def test_int_output():
    def test(x, y, z):
        return x * y * z

    xs = [(torch.rand(4) * 3 + 1).to(torch.int32) for i in range(3)]
    x, y, z = xs
    xn, yn, zn = [t.numpy() for t in xs]
    traced = torch.jit.trace(test, (x, y, z))
    res = traced(x, y, z)
    np.testing.assert_allclose(xn * yn * zn, res.numpy())


def test_abs():
    def easy(x, y):
        c = torch.abs(torch.add(x, y))
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    aa = np.array(1024, dtype=float)
    bb = np.array(1024, dtype=float)
    aa.fill(-0.5)
    bb.fill(-0.5)
    a = torch.from_numpy(aa)
    b = torch.from_numpy(bb)
    x = traced(a, b)
    np.testing.assert_allclose(np.ones(1024), x.numpy())


def test_unary_ops():
    def easy_sin(x, y):
        c = torch.sin(torch.add(x, y))
        return c

    def easy_asin(x, y):
        c = torch.asin(torch.add(x, y))
        return c

    def easy_sinh(x, y):
        c = torch.sinh(torch.add(x, y))
        return c

    def easy_cos(x, y):
        c = torch.cos(torch.add(x, y))
        return c

    def easy_acos(x, y):
        c = torch.acos(torch.add(x, y))
        return c

    def easy_cosh(x, y):
        c = torch.cosh(torch.add(x, y))
        return c

    def easy_tan(x, y):
        c = torch.tan(torch.add(x, y))
        return c

    def easy_atan(x, y):
        c = torch.atan(torch.add(x, y))
        return c

    def easy_tanh(x, y):
        c = torch.tanh(torch.add(x, y))
        return c

    trig_fns = {
        easy_sin: np.sin,
        easy_asin: np.arcsin,
        easy_sinh: np.sinh,
        easy_cos: np.cos,
        easy_acos: np.arccos,
        easy_cosh: np.cosh,
        easy_tan: np.tan,
        easy_atan: np.arctan,
        easy_tanh: np.tanh,
    }

    for torch_fn, np_fn in trig_fns.items():
        traced = torch.jit.trace(torch_fn, (torch.zeros(1024), torch.zeros(1024)))
        aa = np.array(1024, dtype=float)
        bb = np.array(1024, dtype=float)
        aa.fill(0.5)
        bb.fill(0.4)
        a = torch.from_numpy(aa)
        b = torch.from_numpy(bb)
        x = traced(a, b)
        cc = aa + bb
        out = np_fn(cc)
        np.testing.assert_allclose(out, x.numpy())


def test_nans():
    def test_max(x, y):
        return torch.max(2 * x, 2 * y)

    def test_min(x, y):
        return torch.min(2 * x, 2 * y)

    tmax = torch.jit.trace(test_max, (torch.rand(1), torch.rand(1)))
    tmin = torch.jit.trace(test_min, (torch.rand(1), torch.rand(1)))

    x = torch.tensor([np.nan])
    y = torch.tensor([1.0])

    assert(not np.isnan(tmin(x, y).item()))
    assert(np.isnan(tmin(y, x).item()))
    assert(not np.isnan(tmax(x, y).item()))
    assert(np.isnan(tmax(y, x).item()))
