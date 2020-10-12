import copy
import inspect
import typing
import math

import dace
from dace import SDFGState, SDFG, dtypes
from dace.frontend.python.parser import DaceProgram
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs
from dace.registry import autoregister_params
from dace.sdfg.nodes import Node
from dace.symbolic import symstr

from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml.onnx.implementation_abc import ONNXForward
import numpy as np


def program_for_node(program, sdfg: SDFG, state: SDFGState,
                     node: ONNXOp) -> DaceProgram:
    """ Expand a function to a dace program.

        The dtypes for the arguments will be extracted by matching the parameter names to edges.
    """
    input_names = set(inp.name for inp in node.schema.inputs)
    output_names = set(outp.name for outp in node.schema.outputs)

    if input_names.intersection(output_names):
        # this is currently the case for only one onnx op
        raise ValueError(
            "program_for_node cannot be applied on nodes of this type;"
            " '{}' is both an input and an output".format(
                next(input_names.intersection(output_names))))

    params = inspect.signature(program).parameters

    annotations = {}
    for name, param in params.items():
        if name in input_names:
            annotations[name] = node.in_desc_with_name(sdfg, state, name)
        elif name in output_names:
            annotations[name] = node.out_desc_with_name(sdfg, state, name)
        else:
            raise ValueError(
                "'{}' was not found as an input or output for {}".format(
                    name, node.schema.name))

    program.__annotations__ = annotations

    return dace.parser.DaceProgram(program, (), {})


@autoregister_params(op="Sqrt")
class PureSqrt(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return node.in_desc_with_name(sdfg, state, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        def prog(X, Y):
            Y[:] = dace.elementwise(lambda x: sqrt(x), X)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Pow")
class PurePow(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return node.in_desc_with_name(sdfg, state, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        def prog(X, Y, Z):
            Z[:] = X**Y

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Add")
class PureAdd(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        istype = node.in_desc_with_name(sdfg, state, 'A').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

        in_edges = state.in_edges(node)
        input0_dim = len(in_edges[0].data.subset.size())
        input1_dim = len(in_edges[1].data.subset.size())

        if istype and ((input0_dim == 1 and input1_dim == 4) or
                       (input0_dim == 1 and input1_dim <= 3) or
                       (input0_dim <= 3 and input1_dim == 1) or
                       (input0_dim == 3 and input1_dim == 3) or
                       (input0_dim == 2 and input1_dim == 2)):
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
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
                                         'c = a + b')

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

            def prog(A, B, C):
                C[:] = A + B

            return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Sub")
class PureSub(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        istype = node.in_desc_with_name(sdfg, state, 'A').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

        in_edges = state.in_edges(node)
        input0_dim = len(in_edges[0].data.subset.size())
        input1_dim = len(in_edges[1].data.subset.size())

        if istype and ((input0_dim == 1 and input1_dim == 4) or
                       (input0_dim == 1 and input1_dim <= 3) or
                       (input0_dim <= 3 and input1_dim == 1) or
                       (input0_dim == 3 and input1_dim == 3) or
                       (input0_dim == 2 and input1_dim == 2)):
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
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

            def prog(A, B, C):
                C[:] = A - B

            return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Mul")
class PureMul(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        istype = node.in_desc_with_name(sdfg, state, 'A').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

        in_edges = state.in_edges(node)
        input0_dim = len(in_edges[0].data.subset.size())
        input1_dim = len(in_edges[1].data.subset.size())

        if istype and ((input0_dim == 1 and input1_dim == 4) or
                       (input0_dim == 1 and input1_dim <= 3) or
                       (input0_dim <= 3 and input1_dim == 1) or
                       (input0_dim == 3 and input1_dim == 3) or
                       (input0_dim == 2 and input1_dim == 2)):
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
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
                                         'c = a * b')

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

            def prog(A, B, C):
                C[:] = A * B

            return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Div")
class PureDiv(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        istype = node.in_desc_with_name(sdfg, state, 'A').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

        in_edges = state.in_edges(node)
        input0_dim = len(in_edges[0].data.subset.size())
        input1_dim = len(in_edges[1].data.subset.size())

        if istype and ((input0_dim == 1 and input1_dim == 4) or
                       (input0_dim == 1 and input1_dim <= 3) or
                       (input0_dim <= 3 and input1_dim == 1) or
                       (input0_dim == 3 and input1_dim == 3) or
                       (input0_dim == 2 and input1_dim == 2)):
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
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
                                         'c = a / b')

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

            def prog(A, B, C):
                C[:] = A / B

            return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="ReduceMean")
class PureReduceMean(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        axes = None
        keepdims = None
        for name, attr in node.schema.attributes.items():
            if hasattr(node, name):
                if str(node.schema.attributes[name]) == "axes":
                    axes = getattr(node, name)
                elif str(node.schema.attributes[name]) == "keepdims":
                    keepdims = getattr(node, name)

        in_edges = state.in_edges(node)
        input_dim = len(in_edges[0].data.subset.size())

        if input_dim == 3 and axes == [-1] and keepdims == 1:
            return True
        if input_dim == 3 and axes == [2] and keepdims == 1:
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
        sdfg_exp = dace.SDFG('reducemeanExpansion')

        in_edges = state.in_edges(node)

        mm = in_edges[0].data.subset.size()[0]
        nn = in_edges[0].data.subset.size()[1]
        gg = in_edges[0].data.subset.size()[2]

        M = str(mm)
        N = str(nn)
        G = str(gg)

        sdfg_exp.add_array('data', (mm, nn, gg), dace.float32)
        sdfg_exp.add_array('reduced', (mm, nn, 1), dace.float32)
        state_exp = sdfg_exp.add_state()

        me, mx = state_exp.add_map('outer_map', dict(i='0:' + M, j='0:' + N))

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
        sdfg_exp.fill_scope_connectors()

        return sdfg_exp


@autoregister_params(op="Cast")
class PureCast(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
        btype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

        @dace.program
        def castop(input: atype, output: btype):
            #output[:] = int(input)
            output[:] = input

        return castop.to_sdfg()


@autoregister_params(op="Erf")
class PureErf(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return node.in_desc_with_name(sdfg, state, 'input').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        def prog(input, output):
            output[:] = dace.elementwise(lambda x: erf(x), input)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Reshape")
class PureReshape(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:

        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)
        input_dim = len(in_edges[0].data.subset.size())
        output_dim = len(out_edges[0].data.subset.size())

        if input_dim == 4 and output_dim == 3:
            return True
        if input_dim == 3 and output_dim == 4:
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        input_dim = len(in_edges[0].data.subset.size())
        output_dim = len(out_edges[0].data.subset.size())
        sdfg_exp = dace.SDFG('ReshapeExpansion')

        if input_dim == 4 and output_dim == 3:
            ii = in_edges[0].data.subset.size()[0]
            jj = in_edges[0].data.subset.size()[1]
            kk = in_edges[0].data.subset.size()[2]
            ll = in_edges[0].data.subset.size()[3]

            rr = in_edges[1].data.subset.size()[0]

            I = str(ii)
            J = str(jj)
            K = str(kk)
            L = str(ll)

            R = str(rr)

            mm = out_edges[0].data.subset.size()[0]
            nn = out_edges[0].data.subset.size()[1]
            pp = out_edges[0].data.subset.size()[2]

            M = str(mm)
            N = str(nn)
            P = str(pp)

            sdfg_exp.add_array('data', (ii, jj, kk, ll), dace.float32)
            sdfg_exp.add_array('shape', (rr, ), dace.float32)
            sdfg_exp.add_array('reshaped', (mm, nn, pp), dace.float32)

            state_exp = sdfg_exp.add_state()

            task1 = state_exp.add_tasklet('reshape', {'_a', '_dummy'}, {'_b'},
                                          '_b = _a')

            data = state_exp.add_read('data')
            shape = state_exp.add_read('shape')
            reshaped = state_exp.add_access('reshaped')

            me1, mx1 = state_exp.add_map(
                'map1', dict(i='0:' + I, j='0:' + J, k='0:' + K, l='0:' + L))
            state_exp.add_edge(
                data, None, me1, None,
                dace.Memlet.simple(
                    data, '0:' + I + ', 0:' + J + ', 0:' + K + ', 0:' + L))
            state_exp.add_edge(shape, None, me1, None,
                               dace.Memlet.simple(shape, '0:' + R))
            state_exp.add_edge(me1, None, task1, '_a',
                               dace.Memlet.simple(data, 'i, j, k, l'))
            state_exp.add_edge(me1, None, task1, '_dummy',
                               dace.Memlet.simple(shape, '0'))

            state_exp.add_edge(
                task1, '_b', mx1, None,
                dace.Memlet.simple(
                    reshaped,
                    'int((i*{0}*{1}*{2}+j*{1}*{2}+k*{2}+l)/({3}*{4})), \
                     int((i*{0}*{1}*{2}+j*{1}*{2}+k*{2}+l)%({3}*{4})/{4}), \
                     (i*{0}*{1}*{2}+j*{1}*{2}+k*{2}+l)%({3}*{4})%{4}'
                    .format(J, K, L, N, P)))

            state_exp.add_edge(
                mx1, None, reshaped, None,
                dace.Memlet.simple(reshaped,
                                   '0:' + M + ', 0:' + N + ', 0:' + P))

            sdfg_exp.fill_scope_connectors()

        elif input_dim == 3 and output_dim == 4:
            ii = in_edges[0].data.subset.size()[0]
            jj = in_edges[0].data.subset.size()[1]
            kk = in_edges[0].data.subset.size()[2]

            rr = in_edges[1].data.subset.size()[0]

            I = str(ii)
            J = str(jj)
            K = str(kk)

            R = str(rr)

            mm = out_edges[0].data.subset.size()[0]
            nn = out_edges[0].data.subset.size()[1]
            pp = out_edges[0].data.subset.size()[2]
            qq = out_edges[0].data.subset.size()[3]

            M = str(mm)
            N = str(nn)
            P = str(pp)
            Q = str(qq)

            sdfg_exp.add_array('data', (ii, jj, kk), dace.float32)
            sdfg_exp.add_array('shape', (rr, ), dace.float32)
            sdfg_exp.add_array('reshaped', (mm, nn, pp, qq), dace.float32)

            state_exp = sdfg_exp.add_state()

            task1 = state_exp.add_tasklet('reshape', {'_a', '_dummy'}, {'_b'},
                                          '_b = _a')

            data = state_exp.add_read('data')
            shape = state_exp.add_read('shape')
            reshaped = state_exp.add_access('reshaped')

            me1, mx1 = state_exp.add_map(
                'map1', dict(i='0:' + I, j='0:' + J, k='0:' + K))
            state_exp.add_edge(
                data, None, me1, None,
                dace.Memlet.simple(data, '0:' + I + ', 0:' + J + ', 0:' + K))
            state_exp.add_edge(shape, None, me1, None,
                               dace.Memlet.simple(shape, '0:' + R))
            state_exp.add_edge(me1, None, task1, '_a',
                               dace.Memlet.simple(data, 'i, j, k'))
            state_exp.add_edge(me1, None, task1, '_dummy',
                               dace.Memlet.simple(shape, '0'))

            state_exp.add_edge(
                task1, '_b', mx1, None,
                dace.Memlet.simple(
                    reshaped,
                    'int((i*{0}*{1}+j*{1}+k)/({2}*{3}*{4})), \
                     int(((i*{0}*{1}+j*{1}+k)%({2}*{3}*{4}))/({3}*{4})), \
                     int(((i*{0}*{1}+j*{1}+k)%({2}*{3}*{4})%({3}*{4}))/{4}), \
                     (i*{0}*{1}+j*{1}+k)%({2}*{3}*{4})%({3}*{4})%{4}'
                    .format(J, K, N, P, Q)))

            state_exp.add_edge(
                mx1, None, reshaped, None,
                dace.Memlet.simple(
                    reshaped, '0:' + M + ', 0:' + N + ', 0:' + P + ', 0:' + Q))

            sdfg_exp.fill_scope_connectors()
        return sdfg_exp


@autoregister_params(op="MatMul")
class PureMatMul(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        in_edges = state.in_edges(node)
        input0_dim = len(in_edges[0].data.subset.size())
        input1_dim = len(in_edges[1].data.subset.size())

        if input0_dim == 4 and input1_dim == 4:
            return True

        if input0_dim == 3 and input1_dim == 2:
            return True

        if input0_dim == 2 and input1_dim == 2:
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
        in_edges = state.in_edges(node)
        out_edges = state.out_edges(node)

        atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
        btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
        ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

        input0_dim = len(in_edges[0].data.subset.size())
        input1_dim = len(in_edges[1].data.subset.size())
        if input0_dim == 4 and input1_dim == 4:

            @dace.program
            def einsumop(A: atype, B: btype, Y: ctype):
                Y[:] = np.einsum('abik,abkj->abij', A, B)

            return einsumop.to_sdfg()

        if input0_dim == 3 and input1_dim == 2:

            @dace.program
            def einsumop(A: atype, B: btype, Y: ctype):
                Y[:] = np.einsum('bik,kj->bij', A, B)

            return einsumop.to_sdfg()

        if input0_dim == 2 and input1_dim == 2:
            sdfg_exp = dace.SDFG('matmulExpansion')
            ii = in_edges[0].data.subset.size()[0]
            kk = in_edges[0].data.subset.size()[1]
            jj = in_edges[1].data.subset.size()[1]

            I = str(ii)
            K = str(kk)
            J = str(jj)
            sdfg_exp.add_array('A', (ii, kk),
                               sdfg.arrays[in_edges[0].data.data].dtype)
            sdfg_exp.add_array('B', (kk, jj),
                               sdfg.arrays[in_edges[1].data.data].dtype)
            sdfg_exp.add_array('Y', (ii, jj),
                               sdfg.arrays[out_edges[0].data.data].dtype)

            init_state = sdfg_exp.add_state()
            init_state.add_mapped_tasklet(
                'batched_matmul_init', {
                    '_o%d' % i: '0:%s' % symstr(d)
                    for i, d in enumerate((ii, jj))
                }, {},
                'out = 0', {
                    'out':
                    dace.Memlet.simple(
                        'Y', ','.join(
                            ['_o%d' % i for i in range(len((ii, jj)))]))
                },
                external_edges=True)

            state_exp = sdfg_exp.add_state_after(init_state)

            state_exp.add_mapped_tasklet(
                '_MatMult_',
                {'__i%d' % i: '0:%s' % s
                 for i, s in enumerate([I, J, K])}, {
                     '_a': dace.Memlet.simple("A", ('__i0, __i2')),
                     '_b': dace.Memlet.simple("B", ('__i2, __i1'))
                 },
                '_c = _a * _b', {
                    '_c':
                    dace.Memlet.simple(
                        "Y", '__i0, __i1', wcr_str='lambda x, y: x + y')
                },
                external_edges=True)
            return sdfg_exp
            return einsumop.to_sdfg()


@autoregister_params(op="Identity")
class PureIdentity(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        def prog(input, output):
            output[:] = dace.elementwise(lambda x: x, input)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Reciprocal")
class PureReciprocal(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return node.in_desc_with_name(sdfg, state, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)

        def prog(X, Y):
            Y[:] = dace.elementwise(lambda x: 1 / x, X)

        return program_for_node(prog, sdfg, state, node).to_sdfg()


@autoregister_params(op="Tanh")
class PureTanh(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:

        in_edges = state.in_edges(node)
        input_dim = len(in_edges[0].data.subset.size())
        if input_dim == 2:
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
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
        task2 = state_exp.add_tasklet(
            'tanh', {'_a2'}, {'_b2'},
            '_b2 = (exp(_a2) - exp(-_a2))/(exp(_a2) + exp(-_a2))')

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

        me2, mx2 = state_exp.add_map('map2', dict(i='0:' + I, j='0:' + J))
        state_exp.add_edge(tmp_out, None, me2, None,
                           dace.Memlet.simple(tmp_out, '0:' + I + ', 0:' + J))
        state_exp.add_edge(me2, None, task2, '_a2',
                           dace.Memlet.simple(tmp_out, 'i, j'))
        state_exp.add_edge(task2, '_b2', mx2, None,
                           dace.Memlet.simple(output, 'i, j'))
        state_exp.add_edge(mx2, None, output, None,
                           dace.Memlet.simple(output, '0:' + I + ', 0:' + J))
        sdfg_exp.fill_scope_connectors()

        return sdfg_exp


@autoregister_params(op="ReduceSum")
class PureReduceSum(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        axes = None
        keepdims = None
        for name, attr in node.schema.attributes.items():
            if hasattr(node, name):
                if str(node.schema.attributes[name]) == "axes":
                    axes = getattr(node, name)
                elif str(node.schema.attributes[name]) == "keepdims":
                    keepdims = getattr(node, name)

        in_edges = state.in_edges(node)
        input_dim = len(in_edges[0].data.subset.size())

        if input_dim == 4 and axes == [1] and keepdims == 0:
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
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


@autoregister_params(op="Softmax")
class PureSoftmax(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        axis = None
        for name, attr in node.schema.attributes.items():
            if hasattr(node, name):
                if str(node.schema.attributes[name]) == "axis":
                    axis = getattr(node, name)

        in_edges = state.in_edges(node)
        input_dim = len(in_edges[0].data.subset.size())
        if input_dim == 4 and axis == 3:
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
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

        tmp_max = state_exp.add_transient('tmp_max', (1, ), dace.float32)
        tmp_sum = state_exp.add_transient('tmp_sum', (1, ), dace.float32)
        tmp_out = state_exp.add_transient('tmp_out', (ii, jj, kk, ll),
                                          dace.float32)
        input = state_exp.add_read('input')
        output = state_exp.add_access('output')

        red1 = state_exp.add_reduce(
            'lambda a1, b1: max(a1, b1)', None,
            dtypes.min_value(sdfg_exp.arrays['input'].dtype))
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


@autoregister_params(op="Transpose")
class PureTranspose(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        perm = None
        for name, attr in node.schema.attributes.items():
            if hasattr(node, name):
                if str(node.schema.attributes[name]) == "perm":
                    perm = getattr(node, name)

        in_edges = state.in_edges(node)
        input_dim = len(in_edges[0].data.subset.size())

        if input_dim == 4 and perm == [0, 2, 1, 3]:
            return True

        if input_dim == 4 and perm == [0, 2, 3, 1]:
            return True

        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        node.validate(sdfg, state)
        perm = None
        for name, attr in node.schema.attributes.items():
            if hasattr(node, name):
                if str(node.schema.attributes[name]) == "perm":
                    perm = getattr(node, name)

        in_edges = state.in_edges(node)

        sdfg_exp = dace.SDFG('TransposeExpansion')

        ii = in_edges[0].data.subset.size()[0]
        jj = in_edges[0].data.subset.size()[1]
        kk = in_edges[0].data.subset.size()[2]
        ll = in_edges[0].data.subset.size()[3]
        I = str(ii)
        J = str(jj)
        K = str(kk)
        L = str(ll)

        if perm == [0, 2, 1, 3]:
            sdfg_exp.add_array('data', (ii, jj, kk, ll), dace.float32)
            sdfg_exp.add_array('transposed', (ii, kk, jj, ll), dace.float32)

            state_exp = sdfg_exp.add_state()

            task1 = state_exp.add_tasklet('transpose', {'_a'}, {'_b'},
                                          '_b = _a')

            data = state_exp.add_read('data')
            transposed = state_exp.add_access('transposed')

            me1, mx1 = state_exp.add_map(
                'map1', dict(i='0:' + I, j='0:' + J, k='0:' + K, l='0:' + L))
            state_exp.add_edge(
                data, None, me1, None,
                dace.Memlet.simple(
                    data, '0:' + I + ', 0:' + J + ', 0:' + K + ', 0:' + L))
            state_exp.add_edge(me1, None, task1, '_a',
                               dace.Memlet.simple(data, 'i, j, k, l'))
            state_exp.add_edge(task1, '_b', mx1, None,
                               dace.Memlet.simple(transposed, 'i, k, j, l'))
            state_exp.add_edge(
                mx1, None, transposed, None,
                dace.Memlet.simple(
                    transposed,
                    '0:' + I + ', 0:' + K + ', 0:' + J + ', 0:' + L))
            sdfg_exp.fill_scope_connectors()

        elif perm == [0, 2, 3, 1]:
            sdfg_exp.add_array('data', (ii, jj, kk, ll), dace.float32)
            sdfg_exp.add_array('transposed', (ii, kk, ll, jj), dace.float32)

            state_exp = sdfg_exp.add_state()

            task1 = state_exp.add_tasklet('transpose', {'_a'}, {'_b'},
                                          '_b = _a')

            data = state_exp.add_read('data')
            transposed = state_exp.add_access('transposed')

            me1, mx1 = state_exp.add_map(
                'map1', dict(i='0:' + I, j='0:' + J, k='0:' + K, l='0:' + L))
            state_exp.add_edge(
                data, None, me1, None,
                dace.Memlet.simple(
                    data, '0:' + I + ', 0:' + J + ', 0:' + K + ', 0:' + L))
            state_exp.add_edge(me1, None, task1, '_a',
                               dace.Memlet.simple(data, 'i, j, k, l'))
            state_exp.add_edge(task1, '_b', mx1, None,
                               dace.Memlet.simple(transposed, 'i, k, l, j'))
            state_exp.add_edge(
                mx1, None, transposed, None,
                dace.Memlet.simple(
                    transposed,
                    '0:' + I + ', 0:' + K + ', 0:' + L + ', 0:' + J))
            sdfg_exp.fill_scope_connectors()
        return sdfg_exp
