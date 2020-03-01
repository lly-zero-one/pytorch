import numpy as np
import torch


class ExecutionCounter(object):
    def __init__(self, name):
        self.name = name
        self.start_value = torch._C._jit_get_trigger_value(self.name)

    def elapsed_value(self):
        value = torch._C._jit_get_trigger_value(self.name)
        return value - self.start_value


class CudaCodeGenCreated(ExecutionCounter):
    def __init__(self):
        super(CudaCodeGenCreated, self).__init__("cuda_codegen_created")


class CudaCodeGenExecuted(ExecutionCounter):
    def __init__(self):
        super(CudaCodeGenExecuted, self).__init__("cuda_codegen_executed")


class LLVMCodeGenCreated(ExecutionCounter):
    def __init__(self):
        super(LLVMCodeGenCreated, self).__init__("llvm_codegen_created")


class LLVMCodeGenExecuted(ExecutionCounter):
    def __init__(self):
        super(LLVMCodeGenExecuted, self).__init__("llvm_codegen_executed")


class SimpleIREvalExecuted(ExecutionCounter):
    def __init__(self):
        super(SimpleIREvalExecuted, self).__init__("simple_ir_eval_executed")


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
    llvm_executed = LLVMCodeGenExecuted()
    simple_ir_eval_executed = SimpleIREvalExecuted()

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
    assert (
        llvm_executed.elapsed_value() >= 1
        or simple_ir_eval_executed.elapsed_value() >= 1
    )


def test_four_arg():
    def run_addcmul(x, y, z, w):
        c = torch.addcmul(torch.add(x, y), z, w)
        return c

    rand_a = torch.rand(1024, dtype=torch.float)
    rand_b = torch.rand(1024, dtype=torch.float)
    rand_c = torch.rand(1024, dtype=torch.float)
    rand_d = torch.rand(1024, dtype=torch.float)

    traced = torch.jit.trace(
        run_addcmul,
        (
            torch.zeros(1024, dtype=torch.float),
            torch.zeros(1024, dtype=torch.float),
            torch.zeros(1024, dtype=torch.float),
            torch.zeros(1024, dtype=torch.float),
        ),
    )

    x = traced(rand_a, rand_b, rand_c, rand_d)
    y = run_addcmul(rand_a, rand_b, rand_c, rand_d)
    np.testing.assert_allclose(x.numpy(), y.numpy())


def test_three_arg_cuda():
    if not torch.cuda.is_available():
        return
    cuda_cg_executed = CudaCodeGenExecuted()
    cuda_cg_created = CudaCodeGenCreated()

    def test(x, y, z):
        aaa = torch.add(x, y)
        bbb = torch.add(aaa, z)
        return bbb

    M = 32
    N = 32
    traced = torch.jit.trace(
        test,
        (
            torch.rand(M, N, device="cuda"),
            torch.rand(M, N, device="cuda"),
            torch.rand(M, N, device="cuda"),
        ),
    )

    a = torch.rand(M, N, device="cuda")
    b = torch.rand(M, N, device="cuda")
    c = torch.rand(M, N, device="cuda")
    x = traced(a, b, c)
    npr = a.cpu().numpy() + b.cpu().numpy() + c.cpu().numpy()
    np.testing.assert_allclose(npr, x.cpu().numpy())
    assert cuda_cg_executed.elapsed_value() >= 1
    assert cuda_cg_created.elapsed_value() >= 1


def test_broadcast_cuda():
    if not torch.cuda.is_available():
        return

    def test_body(M, N, L, K):
        if not torch.cuda.is_available():
            return
        cuda_cg_executed = CudaCodeGenExecuted()
        cuda_cg_created = CudaCodeGenCreated()

        def test(x, y, z):
            v1 = torch.add(x, y)
            v2 = torch.add(v1, z)
            return v2

        a_shape = [M, N]
        b_shape = [L, M, 1]
        c_shape = [K, L, 1, 1]
        traced = torch.jit.trace(
            test,
            (
                torch.rand(*a_shape, device="cuda"),
                torch.rand(*b_shape, device="cuda"),
                torch.rand(*c_shape, device="cuda"),
            ),
        )

        a = torch.rand(*a_shape, device="cuda")
        b = torch.rand(*b_shape, device="cuda")
        c = torch.rand(*c_shape, device="cuda")
        x = traced(a, b, c)
        npr = a.cpu().numpy() + b.cpu().numpy() + c.cpu().numpy()
        np.testing.assert_allclose(npr, x.cpu().numpy())
        assert cuda_cg_executed.elapsed_value() >= 1
        assert cuda_cg_created.elapsed_value() >= 1

    test_configs = [[36, 17, 63, 33], [32, 32, 32, 32]]
    for test_config in test_configs:
        test_body(*test_config)


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
        traced(a, b), np.maximum(np.minimum(a.numpy(), b.numpy()), [4.0])
    )


def test_clamp():
    def test(x):
        return torch.clamp(x + 3.0, 0.0, 6.0)

    traced = torch.jit.trace(test, (torch.zeros(1024)))
    a = 20.0 * torch.rand(1024) - 10.0
    an = a.numpy()
    np.testing.assert_allclose(traced(a), np.clip(an + 3.0, 0.0, 6.0))


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


def test_unary_ops():
    def test_sin(x, y):
        c = torch.sin(torch.add(x, y))
        return c

    def test_asin(x, y):
        c = torch.asin(torch.add(x, y))
        return c

    def test_sinh(x, y):
        c = torch.sinh(torch.add(x, y))
        return c

    def test_cos(x, y):
        c = torch.cos(torch.add(x, y))
        return c

    def test_acos(x, y):
        c = torch.acos(torch.add(x, y))
        return c

    def test_cosh(x, y):
        c = torch.cosh(torch.add(x, y))
        return c

    def test_tan(x, y):
        c = torch.tan(torch.add(x, y))
        return c

    def test_atan(x, y):
        c = torch.atan(torch.add(x, y))
        return c

    def test_tanh(x, y):
        c = torch.tanh(torch.add(x, y))
        return c

    def test_sqrt(x, y):
        c = torch.sqrt(torch.add(x, y))
        return c

    def test_rsqrt(x, y):
        c = torch.rsqrt(torch.add(x, y))
        return c

    def test_floor(x, y):
        c = torch.floor(torch.add(x, y))
        return c

    def test_ceil(x, y):
        c = torch.ceil(torch.add(x, y))
        return c

    def test_trunc(x, y):
        c = torch.trunc(torch.add(x, y))
        return c

    def test_abs(x, y):
        c = torch.abs(torch.add(x, y))
        return c

    def test_log(x, y):
        c = torch.log(torch.add(x, y))
        return c

    def test_log2(x, y):
        c = torch.log2(torch.add(x, y))
        return c

    def test_log10(x, y):
        c = torch.log10(torch.add(x, y))
        return c

    def test_log1p(x, y):
        c = torch.log1p(torch.add(x, y))
        return c

    def test_rqrt(x, y):
        c = torch.rsqrt(torch.add(x, y))
        return c

    def test_erf(x, y):
        c = torch.erf(torch.add(x, y))
        return c

    def test_exp(x, y):
        c = torch.exp(torch.add(x, y))
        return c

    def test_expm1(x, y):
        c = torch.expm1(torch.add(x, y))
        return c

    def test_erfc(x, y):
        c = torch.erfc(torch.add(x, y))
        return c

    def test_frac(x, y):
        c = torch.frac(torch.add(x, y))
        return c

    def test_lgamma(x, y):
        c = torch.lgamma(torch.add(x, y))
        return c

    def test_sigmoid(x, y):
        c = torch.sigmoid(torch.add(x, y))
        return c

    def test_reciprocal(x, y):
        c = torch.reciprocal(torch.add(x, y))
        return c

    def test_neg(x, y):
        c = torch.neg(torch.add(x, y))
        return c

    def test_relu(x, y):
        c = torch.relu(torch.add(x, y))
        return c

    fns = {
        test_sin,
        test_asin,
        test_sinh,
        test_cos,
        test_acos,
        test_cosh,
        test_tan,
        test_atan,
        test_tanh,
        test_sqrt,
        test_floor,
        test_ceil,
        test_trunc,
        test_abs,
        test_log,
        test_log2,
        test_log10,
        test_log1p,
        test_rsqrt,
        test_exp,
        test_expm1,
        test_erf,
        test_erfc,
        test_frac,
        test_lgamma,
        test_sigmoid,
        test_reciprocal,
        test_neg,
        test_relu,
    }
    rand_a = torch.rand(1024, dtype=torch.float)
    rand_b = torch.rand(1024, dtype=torch.float)
    zeros = torch.zeros(1024, dtype=torch.float)
    cc = np.array(1024, dtype=float)
    cc.fill(np.nan)
    nans = torch.from_numpy(cc)

    for torch_fn in fns:
        # random floats
        traced = torch.jit.trace(
            torch_fn,
            (
                torch.zeros(1024, dtype=torch.float),
                torch.zeros(1024, dtype=torch.float),
            ),
        )
        x = traced(rand_a, rand_b)
        y = torch_fn(rand_a, rand_b)
        np.testing.assert_allclose(x.numpy(), y.numpy(), 1e-7, 1e-6)
        # nans
        traced = torch.jit.trace(torch_fn, (torch.zeros(1024), torch.zeros(1024)))
        x = traced(nans, rand_b)
        y = torch_fn(nans, rand_b)
        np.testing.assert_allclose(x.numpy(), y.numpy())


def test_nans():
    def test_max(x, y):
        return torch.max(2 * x, 2 * y)

    def test_min(x, y):
        return torch.min(2 * x, 2 * y)

    tmax = torch.jit.trace(test_max, (torch.rand(1), torch.rand(1)))
    tmin = torch.jit.trace(test_min, (torch.rand(1), torch.rand(1)))

    x = torch.tensor([np.nan])
    y = torch.tensor([1.0])

    assert not np.isnan(tmin(x, y).item())
    assert np.isnan(tmin(y, x).item())
    assert not np.isnan(tmax(x, y).item())
    assert np.isnan(tmax(y, x).item())


def test_remainder():
    def run_remainder(x, y):
        c = torch.remainder(torch.add(x, y), x)
        return c

    a = torch.rand(1024, dtype=float)
    b = torch.rand(1024, dtype=float)
    zeros = torch.zeros(1024, dtype=float)
    cc = np.array(1024, dtype=float)
    cc.fill(np.nan)
    nans = torch.from_numpy(cc)

    # random floats
    traced = torch.jit.trace(run_remainder, (torch.zeros(1024), torch.zeros(1024)))
    x = traced(a, b)
    y = run_remainder(a, b)
    np.testing.assert_allclose(x.numpy(), y.numpy())

    # div by 0
    traced = torch.jit.trace(run_remainder, (torch.zeros(1024), torch.zeros(1024)))
    x = traced(zeros, a)
    y = run_remainder(zeros, a)
    np.testing.assert_allclose(x.numpy(), y.numpy())

    # numerators and denominatos are nan
    traced = torch.jit.trace(run_remainder, (torch.zeros(1024), torch.zeros(1024)))
    x = traced(nans, a)
    y = run_remainder(nans, a)
    np.testing.assert_allclose(x.numpy(), y.numpy())


def test_multioutput():
    def easy(x):
        b = x + 1
        c = b + b
        return (b, c)

    traced = torch.jit.trace(easy, (torch.zeros(1024)))

    a = torch.zeros(1024)
    b, c = traced(a)
    bp = a.numpy() + 1
    cp = bp + bp
    np.testing.assert_allclose(b.numpy(), bp)
    np.testing.assert_allclose(c.numpy(), cp)


def test_chunk():
    def easy(x):
        y = x + 1
        aaa, bbb = torch.chunk(y, 2)
        return aaa + bbb

    traced = torch.jit.trace(easy, (torch.zeros(1024, 1024)))

    a = torch.zeros(1024, 1024)
    x = traced(a)
    npr = a.numpy()
    npr2 = npr + 1
    npr_a, npr_b = np.array_split(npr2, 2)
    np.testing.assert_allclose(npr_a + npr_b, x.numpy())


def test_cat():
    def easy(x, y):
        a = x + 1
        b = y + 2
        c = torch.cat([a, b], dim=1)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024, 1024), torch.zeros(1024, 1024)))

    a = torch.zeros(1024, 1024)
    x = traced(a, a)
    npr = a.numpy()
    npr_x = npr + 1
    npr_y = npr + 2
    npr_c = np.concatenate((npr_x, npr_y), axis=1)
    np.testing.assert_allclose(npr_c, x.numpy())


def test_scalar():
    @torch.jit.script
    def test_float(x, y, z, a, b):
        # type: (Tensor, Tensor, Tensor, float, float) -> Tensor
        return torch.add(torch.add(x, y, alpha=a), z, alpha=b)

    @torch.jit.script
    def test_int(x, y, z, a, b):
        # type: (Tensor, Tensor, Tensor, int, int) -> Tensor
        return torch.add(torch.add(x, y, alpha=a), z, alpha=b)

    for test in (test_float, test_int):
        llvm = LLVMCodeGenExecuted()
        interp = SimpleIREvalExecuted()
        x, y, z = [torch.rand(4) for i in range(3)]
        a, b = 1, 2
        test(x, y, z, a, b)
        r = test(x, y, z, a, b)
        xn, yn, zn = [t.numpy() for t in (x, y, z)]
        np.testing.assert_allclose(r.numpy(), xn + yn * a + zn * b)
        assert llvm.elapsed_value() == 1 or interp.elapsed_value() == 1

# FIXME: Blocked on profiling executor changes
# def test_loop():
#    @torch.jit.script
#    def test(x, y, z):
#    # type: (Tensor, Tensor, int) -> Tensor
#        b = y
#        for i in range(0, z):
#            a = x + y
#            b = b + y
#        return b
#
#    llvm = LLVMCodeGenExecuted()
#    interp = SimpleIREvalExecuted()
#    x, y, z = (torch.zeros(32, 32), torch.ones(32, 32), 4)
#    test(x, y, z)
#    r = test(x, y, z)
#    assert llvm.elapsed_value == 1 or interp.elapsed_value() == 1
