import numpy as np
import loopy as lp
from mako.template import Template
from pycuda.compiler import SourceModule
from minipyro.codegen.mappers import LoopyMapper


lp_tpl = Template(
    """
rxn_rate[${idx_tuple}] = ${expr}
    """, strict_undefined=True)


def assemble_cuda(ary, knl_name):
    dim = len(ary.shape)
    idx_list = [f'i{i}' for i in range(dim)]

    lp_domains = (
        '{[' + ', '.join(idx_list) + '] : ' +
        ' and '.join([
            f'0 <= i{i} <= {n-1}' for i, n in zip(range(dim), ary.shape)
        ]) + '}'
    )

    lp_mapper = LoopyMapper()
    idx_tuple = ', '.join(idx_list)
    lp_instructions = lp_tpl.render(
        idx_tuple=idx_tuple,
        expr=lp_mapper.rec(ary)
    )

    lp_knl = lp.make_kernel(
        lp_domains, lp_instructions, name=knl_name
    )
    lp_knl = lp.add_dtypes(
        lp_knl,
        {a: np.float64 for a in lp_knl['get_rxn_rate'].arg_dict}
    )

    for i in range(dim):
        lp_knl = lp.split_iname(
            lp_knl, f'i{i}', ary.wg_size,
            outer_tag=f'g.{i}', inner_tag=f'l.{i}'
        )
                                
    lp_knl = lp_knl.copy(target=lp.CudaTarget())
    code_str = lp.generate_code_v2(lp_knl).device_code()
    prg = SourceModule(code_str).get_function(knl_name)
    return prg, code_str
