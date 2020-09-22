import copy
from math import sqrt
import typing

import dace
from dace import SDFGState, SDFG
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs
from dace.registry import autoregister_params
from dace.sdfg.nodes import Node
from dace.symbolic import symstr

from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml.onnx.implementation_abc import ONNXForward


@autoregister_params(op="Div")
class PureDiv(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
        btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
        ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

        @dace.program
        def divop(A: atype, B: btype, C: ctype):
            C[:] = A / B

        return divop.to_sdfg()


@autoregister_params(op="Mul")
class PureMul(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
        btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
        ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

        @dace.program
        def mulop(A: atype, B: btype, C: ctype):
            C[:] = A * B

        return mulop.to_sdfg()


@autoregister_params(op="MatMul")
class PureMatMul(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        in_edges = state.in_edges(node)
        input0_dim = len(in_edges[0].data.subset.size())
        input1_dim = len(in_edges[1].data.subset.size())

        if input0_dim == 2 and input1_dim == 2:
            return True

        if input0_dim == 1 and input1_dim == 2:
            return True

        if input1_dim == 2 and input0_dim == 1:
            return True

        # this can be relaxed
        if input0_dim == 4 and input1_dim == 4:
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
        btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
        ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

        input0_dim = len(in_edges[0].data.subset.size())
        input1_dim = len(in_edges[1].data.subset.size())

        if input0_dim == 4 and input1_dim == 4:
            sdfg_exp = dace.SDFG('matmulExpansion')
            mm = in_edges[0].data.subset.size()[0]
            nn = in_edges[0].data.subset.size()[1]
            ii = in_edges[0].data.subset.size()[2]
            kk = in_edges[0].data.subset.size()[3]
            jj = in_edges[1].data.subset.size()[3]

            M = str(mm)
            N = str(nn)
            I = str(ii)
            K = str(kk)
            J = str(jj)

            sdfg_exp.add_array('A', (mm, nn, ii, kk), dace.float32)
            sdfg_exp.add_array('B', (mm, nn, kk, jj), dace.float32)
            sdfg_exp.add_array('Y', (mm, nn, ii, jj), dace.float32)

            init_state = sdfg_exp.add_state()
            init_state.add_mapped_tasklet(
                'batched_matmul_init', {
                    '_o%d' % i: '0:%s' % symstr(d)
                    for i, d in enumerate((mm, nn, ii, jj))
                }, {},
                'out = 0', {
                    'out':
                    dace.Memlet.simple(
                        'Y', ','.join(
                            ['_o%d' % i
                             for i in range(len((mm, nn, ii, jj)))]))
                },
                external_edges=True)

            state_exp = sdfg_exp.add_state_after(init_state)

            state_exp.add_mapped_tasklet(
                '_BatchedBatchedMatMult_', {
                    '__i%d' % i: '0:%s' % s
                    for i, s in enumerate([M, N, I, J, K])
                }, {
                    '_a': dace.Memlet.simple("A", ('__i0, __i1, __i2, __i4')),
                    '_b': dace.Memlet.simple("B", ('__i0, __i1, __i4, __i3'))
                },
                '_c = _a * _b', {
                    '_c':
                    dace.Memlet.simple("Y",
                                       '__i0, __i1, __i2, __i3',
                                       wcr_str='lambda x, y: x + y')
                },
                external_edges=True)
            return sdfg_exp
        else:

            @dace.program
            def matmultop(A: atype, B: btype, Y: ctype):
                Y[:] = A @ B

            return matmultop.to_sdfg()


@autoregister_params(op="Sub")
class PureSub(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
        btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
        ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

        input0_dim = len(in_edges[0].data.subset.size())
        input1_dim = len(in_edges[1].data.subset.size())

        if input0_dim == 1 and input1_dim == 4:
            mm = in_edges[1].data.subset.size()[0]
            nn = in_edges[1].data.subset.size()[1]
            gg = in_edges[1].data.subset.size()[2]
            hh = in_edges[1].data.subset.size()[3]

            M = str(mm)
            N = str(nn)
            G = str(gg)
            H = str(hh)

            sdfg_exp = dace.SDFG('subExpansion')
            sdfg_exp.add_array('A', (1, ), dace.float32)
            sdfg_exp.add_array('B', (mm, nn, gg, hh), dace.float32)
            sdfg_exp.add_array('C', (mm, nn, gg, hh), dace.float32)
            state_exp = sdfg_exp.add_state()

            me, mx = state_exp.add_map(
                'outer_map',
                dict(i='0:' + M, j='0:' + N, k='0:' + G, l='0:' + H))

            A = state_exp.add_read('A')
            B = state_exp.add_read('B')
            C = state_exp.add_access('C')
            texp = state_exp.add_tasklet('tasklet', {'a', 'b'}, {'c'},
                                         'c = a - b')

            state_exp.add_edge(A, None, me, None, dace.Memlet.simple(A, '0'))
            state_exp.add_edge(
                B, None, me, None,
                dace.Memlet.simple(
                    B, '0:' + M + ', 0:' + N + ', 0:' + G + ', 0:' + H))
            state_exp.add_edge(me, None, texp, "a", dace.Memlet.simple(A, '0'))
            state_exp.add_edge(me, None, texp, "b",
                               dace.Memlet.simple(B, 'i, j, k, l'))
            state_exp.add_edge(texp, "c", mx, None,
                               dace.Memlet.simple(C, 'i, j, k, l'))
            state_exp.add_edge(
                mx, None, C, None,
                dace.Memlet.simple(
                    C, '0:' + M + ', 0:' + N + ', 0:' + G + ', 0:' + H))

            sdfg_exp.fill_scope_connectors()
            return sdfg_exp
        else:

            @dace.program
            def subop(A: atype, B: btype, C: ctype):
                C[:] = A - B

            return subop.to_sdfg()


@autoregister_params(op="Add")
class PureAdd(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)

        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
        btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
        ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

        @dace.program
        def powop(X: atype, Y: btype, Z: ctype):
            Z[:] = X**Y

        return powop.to_sdfg()


@autoregister_params(op="Identity")
class PureIdentity(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        # in_edges = state.in_edges(node)
        # out_edges = state.out_edges(node)

        # atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
        # btype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

        # @dace.program
        # def idop(input: atype, output: btype):
        #    output[:] = input
        # return idop.to_sdfg()

        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        input_dim = len(in_edges[0].data.subset.size())
        output_dim = len(out_edges[0].data.subset.size())

        sdfg_exp = dace.SDFG('idenExpansion')

        if input_dim == 3 and output_dim == 3:

            mm = in_edges[0].data.subset.size()[0]
            nn = in_edges[0].data.subset.size()[1]
            kk = in_edges[0].data.subset.size()[2]

            M = str(mm)
            N = str(nn)
            K = str(kk)

            sdfg_exp.add_array('input', (mm, nn, kk), dace.float32)
            sdfg_exp.add_array('output', (mm, nn, kk), dace.float32)

            state_exp = sdfg_exp.add_state()
            input = state_exp.add_read('input')
            output = state_exp.add_access('output')
            me, mx = state_exp.add_map(
                'outer_map', dict(i='0:' + M, j='0:' + N, k='0:' + K))

            tiden = state_exp.add_tasklet('tiden', {'_a'}, {'_b'}, '_b = _a')

            state_exp.add_edge(
                input, None, me, None,
                dace.Memlet.simple(input, '0:' + M + ', 0:' + N + ', 0:' + K))
            state_exp.add_edge(me, None, tiden, '_a',
                               dace.Memlet.simple(input, 'i, j, k'))
            state_exp.add_edge(tiden, '_b', mx, None,
                               dace.Memlet.simple(output, 'i, j, k'))
            state_exp.add_edge(
                mx, None, output, None,
                dace.Memlet.simple(output, '0:' + M + ', 0:' + N + ', 0:' + K))
            sdfg_exp.fill_scope_connectors()
            return sdfg_exp
        elif input_dim == 2 and output_dim == 2:
            mm = in_edges[0].data.subset.size()[0]
            nn = in_edges[0].data.subset.size()[1]

            M = str(mm)
            N = str(nn)

            sdfg_exp.add_array('input', (mm, nn), dace.float32)
            sdfg_exp.add_array('output', (mm, nn), dace.float32)

            state_exp = sdfg_exp.add_state()
            input = state_exp.add_read('input')
            output = state_exp.add_access('output')
            me, mx = state_exp.add_map('outer_map', dict(i='0:' + M,
                                                         j='0:' + N))

            tiden = state_exp.add_tasklet('tiden', {'_a'}, {'_b'}, '_b = _a')

            state_exp.add_edge(
                input, None, me, None,
                dace.Memlet.simple(input, '0:' + M + ', 0:' + N))
            state_exp.add_edge(me, None, tiden, '_a',
                               dace.Memlet.simple(input, 'i, j'))
            state_exp.add_edge(tiden, '_b', mx, None,
                               dace.Memlet.simple(output, 'i, j'))
            state_exp.add_edge(
                mx, None, output, None,
                dace.Memlet.simple(output, '0:' + M + ', 0:' + N))
            sdfg_exp.fill_scope_connectors()
            return sdfg_exp


@autoregister_params(op="Reciprocal")
class PureReciprocal(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
        btype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

        @dace.program
        def recop(X: atype, Y: btype):
            Y[:] = 1 / X

        return recop.to_sdfg()


@autoregister_params(op="Sqrt")
class PureSqrt(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)

        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
        btype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

        @dace.program
        def sqrtop(X: atype, Y: btype):
            # Y[:] = X ** dace.float32(0.5)
            Y[:] = sqrt(X)

        return sqrtop.to_sdfg()


@autoregister_params(op="Tanh")
class PureTanh(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)

        in_edges = state.in_edges(node)

        ii = in_edges[0].data.subset.size()[0]
        jj = in_edges[0].data.subset.size()[1]

        I = str(ii)
        J = str(jj)

        sdfg_exp = dace.SDFG('tanhExpansion')
        sdfg_exp.add_array('input', (ii, jj), dace.float32)
        sdfg_exp.add_array('output', (ii, jj), dace.float32)

        state_exp = sdfg_exp.add_state()

        tmp_out = state_exp.add_transient('tmp_out', (ii, jj), dace.float32)

        task1 = state_exp.add_tasklet(
            'threshold1', {'_a1'}, {'_b1'},
            '_b1 = 80.0 if _a1 > 80.0 else (-80.0 if _a1 < -80.0 else _a1)')
        # task2 = state_exp.add_tasklet('threshold2', {'_a2'}, {'_b2'}, '_b2 = -80.0 if _a2 < -80.0 else _a2')
        task3 = state_exp.add_tasklet(
            'tanh', {'_a3'}, {'_b3'},
            '_b3 = (exp(_a3) - exp(-_a3))/(exp(_a3) + exp(-_a3))')

        input = state_exp.add_read('input')
        output = state_exp.add_access('output')

        me1, mx1 = state_exp.add_map('map1', dict(i='0:' + I, j='0:' + J))
        state_exp.add_edge(input, None, me1, None,
                           dace.Memlet.simple(input, '0:' + I + ', 0:' + J))
        state_exp.add_edge(me1, None, task1, '_a1',
                           dace.Memlet.simple(input, 'i, j'))
        state_exp.add_edge(task1, '_b1', mx1, None,
                           dace.Memlet.simple(tmp_out, 'i, j'))
        state_exp.add_edge(mx1, None, tmp_out, None,
                           dace.Memlet.simple(tmp_out, '0:' + I + ', 0:' + J))

        # me2, mx2 = state_exp.add_map('map2', dict(i='0:' + I, j='0:' + J))
        # state_exp.add_edge(output, None, me2, None, dace.Memlet.simple(output, '0:'+I+', 0:'+J))
        # state_exp.add_edge(me2, None, task2, '_a2', dace.Memlet.simple(output, 'i, j'))
        # state_exp.add_edge(task2, '_b2', mx2, None, dace.Memlet.simple(output, 'i, j'))
        # state_exp.add_edge(mx2, None, output, None, dace.Memlet.simple(output, '0:'+I+', 0:'+J))

        me3, mx3 = state_exp.add_map('map3', dict(i='0:' + I, j='0:' + J))
        state_exp.add_edge(tmp_out, None, me3, None,
                           dace.Memlet.simple(tmp_out, '0:' + I + ', 0:' + J))
        state_exp.add_edge(me3, None, task3, '_a3',
                           dace.Memlet.simple(tmp_out, 'i, j'))
        state_exp.add_edge(task3, '_b3', mx3, None,
                           dace.Memlet.simple(output, 'i, j'))
        state_exp.add_edge(mx3, None, output, None,
                           dace.Memlet.simple(output, '0:' + I + ', 0:' + J))
        sdfg_exp.fill_scope_connectors()

        return sdfg_exp


@autoregister_params(op="ReduceSum")
class PureReduceSum(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        in_edges = state.in_edges(node)
        mm = in_edges[0].data.subset.size()[0]
        nn = in_edges[0].data.subset.size()[1]
        gg = in_edges[0].data.subset.size()[2]
        hh = in_edges[0].data.subset.size()[3]

        M = str(mm)
        N = str(nn)
        G = str(gg)
        H = str(hh)

        sdfg_exp = dace.SDFG('reducesumExpansion')
        sdfg_exp.add_array('data', (mm, nn, gg, hh), dace.float32)
        sdfg_exp.add_array('reduced', (mm, gg, hh), dace.float32)
        state_exp = sdfg_exp.add_state()

        me, mx = state_exp.add_map('outer_map',
                                   dict(i='0:' + M, k='0:' + G, l='0:' + H))

        data = state_exp.add_read('data')
        reduced = state_exp.add_access('reduced')

        redsum = state_exp.add_reduce('lambda a1, b1: a1 + b1', None, 0)
        tmp_sum = state_exp.add_transient('tmp_sum', (1, ), dace.float32)

        state_exp.add_edge(
            data, None, me, None,
            dace.Memlet.simple(data, '0:' + M + ', 0:' + N + ', 0:' + G +
                               ', 0:' + H))
        state_exp.add_edge(me, None, redsum, None,
                           dace.Memlet.simple(data, 'i, 0:' + N + ', k, l'))
        state_exp.add_edge(redsum, None, tmp_sum, None,
                           dace.Memlet.simple(tmp_sum, '0'))
        state_exp.add_edge(tmp_sum, None, mx, None,
                           dace.Memlet.simple(reduced, 'i, k, l'))
        state_exp.add_edge(
            mx, None, reduced, None,
            dace.Memlet.simple(reduced, '0:' + M + ', 0:' + G + ', 0:' + H))

        sdfg_exp.fill_scope_connectors()

        return sdfg_exp


@autoregister_params(op="ReduceMean")
class PureReduceMean(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
        axes = None
        keepdims = None
        for name, attr in node.schema.attributes.items():
            if hasattr(node, name):
                if str(node.schema.attributes[name]) == "axes":
                    axes = getattr(node, name)
                elif str(node.schema.attributes[name]) == "keepdims":
                    keepdims = getattr(node, name)

        sdfg_exp = dace.SDFG('reducemeanExpansion')

        in_edges = state.in_edges(node)

        if int(axes[0]) == 2:
            mm = in_edges[0].data.subset.size()[0]
            nn = in_edges[0].data.subset.size()[1]
            gg = in_edges[0].data.subset.size()[2]

            M = str(mm)
            N = str(nn)
            G = str(gg)

            sdfg_exp.add_array('data', (mm, nn, gg), dace.float32)
            sdfg_exp.add_array('reduced', (mm, nn, 1), dace.float32)
            state_exp = sdfg_exp.add_state()

            me, mx = state_exp.add_map('outer_map', dict(i='0:' + M,
                                                         j='0:' + N))

            data = state_exp.add_read('data')
            reduced = state_exp.add_access('reduced')

            redsum = state_exp.add_reduce('lambda a1, b1: a1 + b1', None, 0)
            tmp_sum = state_exp.add_transient('tmp_sum', (1, ), dace.float32)

            tmean = state_exp.add_tasklet('meantasklet', {'tsum'}, {'mean'},
                                          'mean = tsum / (%s)' % G)

            state_exp.add_edge(
                data, None, me, None,
                dace.Memlet.simple(data, '0:' + M + ', 0:' + N + ', 0:' + G))
            state_exp.add_edge(me, None, redsum, None,
                               dace.Memlet.simple(data, 'i, j, 0:' + G))
            state_exp.add_edge(redsum, None, tmp_sum, None,
                               dace.Memlet.simple(tmp_sum, '0'))
            state_exp.add_edge(tmp_sum, None, tmean, 'tsum',
                               dace.Memlet.simple(tmp_sum, '0'))
            state_exp.add_edge(tmean, 'mean', mx, None,
                               dace.Memlet.simple(reduced, 'i, j, 0'))
            state_exp.add_edge(
                mx, None, reduced, None,
                dace.Memlet.simple(reduced, '0:' + M + ', 0:' + N + ', 0'))

        elif int(axes[0]) == 1:
            mm = in_edges[0].data.subset.size()[0]
            nn = in_edges[0].data.subset.size()[1]

            M = str(mm)
            N = str(nn)

            sdfg_exp.add_array('data', (mm, nn), dace.float32)
            sdfg_exp.add_array('reduced', (mm, 1), dace.float32)
            state_exp = sdfg_exp.add_state()

            me, mx = state_exp.add_map('outer_map', dict(i='0:' + M))

            data = state_exp.add_read('data')
            reduced = state_exp.add_access('reduced')

            redsum = state_exp.add_reduce('lambda a1, b1: a1 + b1', None, 0)
            tmp_sum = state_exp.add_transient('tmp_sum', (1, ), dace.float32)
            tmean = state_exp.add_tasklet('meantasklet', {'tsum'}, {'mean'},
                                          'mean = tsum / (%s)' % N)

            state_exp.add_edge(data, None, me, None,
                               dace.Memlet.simple(data, '0:' + M + ', 0:' + N))
            state_exp.add_edge(me, None, redsum, None,
                               dace.Memlet.simple(data, 'i, 0:' + N))
            state_exp.add_edge(redsum, None, tmp_sum, None,
                               dace.Memlet.simple(tmp_sum, '0'))
            state_exp.add_edge(tmp_sum, None, tmean, 'tsum',
                               dace.Memlet.simple(tmp_sum, '0'))
            state_exp.add_edge(tmean, 'mean', mx, None,
                               dace.Memlet.simple(reduced, 'i, 0'))
            state_exp.add_edge(mx, None, reduced, None,
                               dace.Memlet.simple(reduced, '0:' + M + ', 0'))

        sdfg_exp.fill_scope_connectors()

        return sdfg_exp


@autoregister_params(op="Softmax")
class PureSoftmax(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        axis = None
        for name, attr in node.schema.attributes.items():
            if hasattr(node, name):
                if str(node.schema.attributes[name]) == "axis":
                    axis = getattr(node, name)
        assert (axis == 3)
        in_edges = state.in_edges(node)
        ii = in_edges[0].data.subset.size()[0]
        jj = in_edges[0].data.subset.size()[1]
        kk = in_edges[0].data.subset.size()[2]
        ll = in_edges[0].data.subset.size()[3]
        I = str(ii)
        J = str(jj)
        K = str(kk)
        L = str(ll)
        sdfg_exp = dace.SDFG('softmaxExpansion')
        sdfg_exp.add_array('input', (ii, jj, kk, ll), dace.float32)
        sdfg_exp.add_array('output', (ii, jj, kk, ll), dace.float32)
        state_exp = sdfg_exp.add_state()
        ome, omx = state_exp.add_map('outer_map',
                                     dict(i='0:' + I, j='0:' + J, k='0:' + K))
        ime, imx = state_exp.add_map('inner_map', dict(l='0:' + L))

        # tmp_max = dace.define_local([1], dtype=dace.float32)
        # tmp_sum = dace.define_local([1], dtype=dace.float32)
        tmp_max = state_exp.add_transient('tmp_max', (1, ), dace.float32)
        tmp_sum = state_exp.add_transient('tmp_sum', (1, ), dace.float32)
        tmp_out = state_exp.add_transient('tmp_out', (ii, jj, kk, ll),
                                          dace.float32)
        input = state_exp.add_read('input')
        output = state_exp.add_access('output')

        red1 = state_exp.add_reduce('lambda a1, b1: max(a1, b1)', None, 0)
        texp1 = state_exp.add_tasklet('tasklet1', {'a2', 'b2'}, {'c2'},
                                      'c2 = exp(a2-b2)')

        state_exp.add_edge(
            input, None, ome, None,
            dace.Memlet.simple(input, '0:' + I + ', 0:' + J + ', 0:' + K +
                               ', 0:' + L))
        state_exp.add_edge(ome, None, red1, None,
                           dace.Memlet.simple(input, 'i, j, k, 0:' + L))
        state_exp.add_edge(red1, None, tmp_max, None,
                           dace.Memlet.simple(tmp_max, '0'))

        state_exp.add_edge(ome, None, ime, None,
                           dace.Memlet.simple(input, 'i, j, k, 0:' + L))
        state_exp.add_edge(tmp_max, None, ime, None,
                           dace.Memlet.simple(tmp_max, '0'))

        state_exp.add_edge(ime, None, texp1, "a2",
                           dace.Memlet.simple(input, 'i, j, k, l'))
        state_exp.add_edge(ime, None, texp1, "b2",
                           dace.Memlet.simple(tmp_max, '0'))
        state_exp.add_edge(texp1, "c2", imx, None,
                           dace.Memlet.simple(tmp_out, 'i, j, k, l'))
        state_exp.add_edge(imx, None, omx, None,
                           dace.Memlet.simple(tmp_out, 'i, j, k, 0:' + L))
        state_exp.add_edge(
            omx, None, tmp_out, None,
            dace.Memlet.simple(tmp_out, '0:' + I + ', 0:' + J + ', 0:' + K +
                               ', 0:' + L))

        ome1, omx1 = state_exp.add_map(
            'outer_map1', dict(i='0:' + I, j='0:' + J, k='0:' + K))
        ime1, imx1 = state_exp.add_map('inner_map1', dict(l='0:' + L))
        red2 = state_exp.add_reduce('lambda a3, b3: a3 + b3', None, 0)
        texp2 = state_exp.add_tasklet('tasklet2', {'a4', 'b4'}, {'c4'},
                                      'c4 = a4 / b4')

        state_exp.add_edge(
            tmp_out, None, ome1, None,
            dace.Memlet.simple(tmp_out, '0:' + I + ', 0:' + J + ', 0:' + K +
                               ', 0:' + L))
        state_exp.add_edge(ome1, None, red2, None,
                           dace.Memlet.simple(tmp_out, 'i, j, k, 0:' + L))
        state_exp.add_edge(red2, None, tmp_sum, None,
                           dace.Memlet.simple(tmp_sum, '0'))

        state_exp.add_edge(ome1, None, ime1, None,
                           dace.Memlet.simple(tmp_out, 'i, j, k, 0:' + L))
        state_exp.add_edge(tmp_sum, None, ime1, None,
                           dace.Memlet.simple(tmp_sum, '0'))

        state_exp.add_edge(ime1, None, texp2, "a4",
                           dace.Memlet.simple(tmp_out, 'i, j, k, l'))
        state_exp.add_edge(ime1, None, texp2, "b4",
                           dace.Memlet.simple(tmp_sum, '0'))
        state_exp.add_edge(texp2, "c4", imx1, None,
                           dace.Memlet.simple(output, 'i, j, k, l'))
        state_exp.add_edge(imx1, None, omx1, None,
                           dace.Memlet.simple(output, 'i, j, k, 0:' + L))
        state_exp.add_edge(
            omx1, None, output, None,
            dace.Memlet.simple(output, '0:' + I + ', 0:' + J + ', 0:' + K +
                               ', 0:' + L))

        sdfg_exp.fill_scope_connectors()

        return sdfg_exp
