import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.ascend.language import vmi as V


@tilelang.jit(target='pto')
def get_engram_sinkhorn_momentum_kernel_vmi(
    grad_dtype: T.dtype,
    momentum_dtype: T.dtype,
    nesterov: bool,
    beta1: float,
    num_vec_cores: int,
):
    vec_size = 64
    row_size = 256
    rows_per_tile = 16
    num_stages = 4

    num_rows = T.dynamic('num_rows')
    num_chunks = row_size // vec_size
    grad_is_bf16 = grad_dtype == T.bfloat16
    momentum_is_bf16 = momentum_dtype == T.bfloat16

    @T.prim_func
    def engram_sinkhorn_momentum_kernel_vmi(
        grad: T.Tensor[(num_rows, row_size), grad_dtype],
        momentum_buffer: T.Tensor[(num_rows, row_size), momentum_dtype],
        sinkhorn_input: T.Tensor[(num_rows, row_size), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            grad_ub = T.alloc_shared((rows_per_tile, row_size), grad_dtype)
            momentum_ub = T.alloc_shared((rows_per_tile, row_size), momentum_dtype)
            sinkhorn_ub = T.alloc_shared((rows_per_tile, row_size), T.float32)
            T.annotate_buffer_versions(
                {
                    grad_ub: num_stages,
                    momentum_ub: num_stages,
                    sinkhorn_ub: num_stages,
                }
            )

            for tile_id in T.Persistent(
                [T.ceildiv(num_rows, rows_per_tile)],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                row_start = tile_id * rows_per_tile
                T.copy(grad[row_start : row_start + rows_per_tile, :], grad_ub)
                T.copy(momentum_buffer[row_start : row_start + rows_per_tile, :], momentum_ub)

                with T.SimdVF():
                    mask = V.create_mask(vec_size, size=vec_size)
                    beta_vec = V.vbrc(T.float32(beta1), size=vec_size)
                    for row in T.serial(rows_per_tile):
                        for chunk in T.unroll(num_chunks, explicit=True):
                            col = chunk * vec_size
                            if grad_is_bf16:
                                grad_f32 = V.vcvt(V.vload(grad_ub[row, col], size=vec_size), 'float32')
                            else:
                                grad_f32 = V.vload(grad_ub[row, col], size=vec_size)
                            if momentum_is_bf16:
                                momentum_f32 = V.vcvt(V.vload(momentum_ub[row, col], size=vec_size), 'float32')
                            else:
                                momentum_f32 = V.vload(momentum_ub[row, col], size=vec_size)

                            scaled_grad = V.vmuls(grad_f32, T.float32(1.0 - beta1), mask)
                            momentum_f32 = V.vmula(scaled_grad, beta_vec, momentum_f32, mask)
                            if momentum_is_bf16:
                                V.vstore(V.vcvt(momentum_f32, 'bfloat16'), momentum_ub[row, col])
                            else:
                                V.vstore(momentum_f32, momentum_ub[row, col])

                            if nesterov:
                                scaled_grad = V.vaxpy(momentum_f32, scaled_grad, T.float32(beta1), mask)
                                V.vstore(scaled_grad, sinkhorn_ub[row, col])
                            else:
                                V.vstore(momentum_f32, sinkhorn_ub[row, col])

                T.copy(momentum_ub, momentum_buffer[row_start : row_start + rows_per_tile, :])
                T.copy(sinkhorn_ub, sinkhorn_input[row_start : row_start + rows_per_tile, :])

    return engram_sinkhorn_momentum_kernel_vmi


_STEP_PTO_FLAGS = ['--enable-vpto-soft-postupdate=false']
@tilelang.jit(target='pto', compile_flags=_STEP_PTO_FLAGS)
def get_engram_sinkhorn_step_kernel_vmi(
    head_dim: int,
    eps: float,
    num_vec_cores: int,
):
    vec_size = 64
    num_accs = 4
    assert head_dim % (vec_size * num_accs) == 0
    num_rows = T.dynamic('num_rows')
    num_chunks = head_dim // vec_size
    num_stages = 4
    batch_size = 256

    @T.prim_func
    def engram_sinkhorn_step_kernel_vmi(
        matrix: T.Tensor[(num_rows, head_dim), T.float32],
        row_scale: T.Tensor[(num_rows,), T.float32],
        col_scale: T.Tensor[(head_dim,), T.float32],
        row_norm: T.Tensor[(num_rows,), T.float32],
        col_sumsq_partial: T.Tensor[(num_vec_cores, head_dim), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            col_scale_ub = T.alloc_shared((head_dim,), T.float32)
            col_sumsq_ub = T.alloc_shared((head_dim,), T.float32)
            matrix_ub = T.alloc_shared((head_dim,), T.float32)
            row_stats_batch_ub = T.alloc_shared((2, batch_size), T.float32)
            T.annotate_buffer_versions(
                {
                    matrix_ub: num_stages,
                    row_stats_batch_ub: num_stages,
                }
            )

            T.copy(col_scale, col_scale_ub)
            with T.SimdVF():
                T.clear(col_sumsq_ub)

            num_batches = T.ceildiv(num_rows, batch_size)
            for batch_id in T.Persistent([num_batches], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
                row_start = batch_id * batch_size
                batch_rows = T.min(batch_size, num_rows - row_start)
                T.copy(row_scale[row_start : row_start + batch_rows], row_stats_batch_ub[0, :batch_rows])
                for output_slot in T.serial(batch_rows):
                    row = row_start + output_slot
                    T.copy(matrix[row, :], matrix_ub)

                    with T.SimdVF():
                        mask = V.create_mask(vec_size, size=vec_size)
                        f32 = V.vreg(vec_size, T.float32)
                        zero = V.vbrc(T.float32(0.0), size=vec_size)
                        accum = V.alloc_local((num_accs,), f32)
                        for acc_id in T.unroll(num_accs, explicit=True):
                            accum[acc_id] = zero

                        scale1 = V.vload(row_stats_batch_ub[0, output_slot], size=1)
                        scale = V.vbrc(scale1, size=vec_size)
                        for chunk_group in T.serial(num_chunks // num_accs):
                            for acc_id in T.unroll(num_accs, explicit=True):
                                chunk = chunk_group * num_accs + acc_id
                                offset = chunk * vec_size
                                base = V.vmul(V.vmul(scale, V.vload(matrix_ub[offset], size=vec_size)), V.vload(col_scale_ub[offset], size=vec_size))
                                accum[acc_id] = V.vmula(accum[acc_id], base, base, mask)
                                V.vstore(base, matrix_ub[offset])

                        for acc_id in T.unroll(2, explicit=True):
                            accum[acc_id] = V.vadd(accum[acc_id], accum[acc_id + 2])
                        accum[0] = V.vadd(accum[0], accum[1])

                        mask1 = V.create_mask(1, size=1)
                        one1 = V.vbrc(T.float32(1.0), size=1)
                        norm = V.vsqrt(V.vcadd(accum[0], mask, reassoc=True))
                        inv_norm = V.vdiv(one1, V.vadds(norm, T.float32(eps), mask1), mask1, precision='exact')
                        normalized_scale = V.vmul(scale1, inv_norm)
                        inv_norm_vec = V.vbrc(inv_norm, size=vec_size)
                        V.vstore(norm, row_stats_batch_ub[1, output_slot])
                        V.vstore(normalized_scale, row_stats_batch_ub[0, output_slot])

                        S.mem_bar('VST_VLD')
                        for chunk in T.serial(num_chunks):
                            offset = chunk * vec_size
                            col_acc = V.vload(col_sumsq_ub[offset], size=vec_size)
                            normalized_base = V.vmul(V.vload(matrix_ub[offset], size=vec_size), inv_norm_vec)
                            col_acc = V.vmula(col_acc, normalized_base, normalized_base, mask)
                            V.vstore(col_acc, col_sumsq_ub[offset])

                # TODO: tilelang bugfix
                with T.Task():
                    T.copy(row_stats_batch_ub[1, :batch_rows], row_norm[row_start : row_start + batch_rows])
                    T.copy(row_stats_batch_ub[0, :batch_rows], row_scale[row_start : row_start + batch_rows])

            T.ascend_pipe_barrier('PIPE_ALL')
            T.copy(col_sumsq_ub, col_sumsq_partial[core_id, :])

    return engram_sinkhorn_step_kernel_vmi


@tilelang.jit(target='pto')
def get_engram_sinkhorn_step_reduce_kernel_vmi(
    head_dim: int,
    num_partials: int,
    num_vec_cores: int,
):
    vec_size = 64
    num_accs = 4
    blk_d = vec_size * num_accs
    assert head_dim % blk_d == 0
    num_col_blocks = head_dim // blk_d
    reduce_cores = min(num_vec_cores, num_col_blocks)

    @T.prim_func
    def engram_sinkhorn_step_reduce_kernel_vmi(
        col_sumsq_partial: T.Tensor[(num_partials, head_dim), T.float32],
        col_sumsq: T.Tensor[(head_dim,), T.float32],
    ):
        with T.Kernel(reduce_cores) as core_id:
            partial_ub = T.alloc_shared((num_partials, blk_d), T.float32)
            col_sumsq_ub = T.alloc_shared((blk_d,), T.float32)

            for col_block in T.Persistent([num_col_blocks], reduce_cores, core_id, group_size=1, num_stages=0):
                col_start = col_block * blk_d
                T.copy(col_sumsq_partial[:, col_start : col_start + blk_d], partial_ub)

                with T.SimdVF():
                    f32 = V.vreg(vec_size, T.float32)
                    zero = V.vbrc(T.float32(0.0), size=vec_size)
                    accum = V.alloc_local((num_accs,), f32)
                    for acc_id in T.unroll(num_accs, explicit=True):
                        accum[acc_id] = zero

                    for partial_id in T.serial(num_partials):
                        for acc_id in T.unroll(num_accs, explicit=True):
                            offset = acc_id * vec_size
                            accum[acc_id] = V.vadd(accum[acc_id], V.vload(partial_ub[partial_id, offset], size=vec_size))

                    for acc_id in T.unroll(num_accs, explicit=True):
                        V.vstore(accum[acc_id], col_sumsq_ub[acc_id * vec_size])

                T.copy(col_sumsq_ub, col_sumsq[col_start : col_start + blk_d])

    return engram_sinkhorn_step_reduce_kernel_vmi


@tilelang.jit(target='pto')
def get_engram_sinkhorn_finalize_kernel_vmi(
    head_dim: int,
    eps: float,
    alpha: float,
    num_vec_cores: int,
):
    vec_size = 64
    num_accs = 4
    assert head_dim % (vec_size * num_accs) == 0
    num_rows = T.dynamic('num_rows')
    num_chunks = head_dim // vec_size
    num_stages = 4

    @T.prim_func
    def engram_sinkhorn_finalize_kernel_vmi(
        matrix: T.Tensor[(num_rows, head_dim), T.float32],
        row_scale: T.Tensor[(num_rows,), T.float32],
        col_scale: T.Tensor[(head_dim,), T.float32],
        out: T.Tensor[(num_rows, head_dim), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            col_scale_ub = T.alloc_shared((head_dim,), T.float32)
            matrix_ub = T.alloc_shared((head_dim,), T.float32)
            out_ub = T.alloc_shared((head_dim,), T.float32)
            row_scale_ub = T.alloc_shared((8,), T.float32)
            T.annotate_buffer_versions(
                {
                    matrix_ub: num_stages,
                    out_ub: num_stages,
                    row_scale_ub: num_stages,
                }
            )

            T.copy(col_scale, col_scale_ub)
            for row in T.Persistent([num_rows], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
                T.copy(matrix[row, :], matrix_ub)
                T.copy(row_scale[row : row + 1], row_scale_ub[:1])

                with T.SimdVF():
                    mask = V.create_mask(vec_size, size=vec_size)
                    f32 = V.vreg(vec_size, T.float32)
                    zero = V.vbrc(T.float32(0.0), size=vec_size)
                    accum = V.alloc_local((num_accs,), f32)
                    for acc_id in T.unroll(num_accs, explicit=True):
                        accum[acc_id] = zero

                    scale = V.vbrc(V.vload(row_scale_ub[0], size=1), size=vec_size)
                    for chunk_group in T.serial(num_chunks // num_accs):
                        for acc_id in T.unroll(num_accs, explicit=True):
                            offset = (chunk_group * num_accs + acc_id) * vec_size
                            value = V.vmul(V.vmul(scale, V.vload(matrix_ub[offset], size=vec_size)), V.vload(col_scale_ub[offset], size=vec_size))
                            accum[acc_id] = V.vmula(accum[acc_id], value, value, mask)
                            V.vstore(value, out_ub[offset])

                    for acc_id in T.unroll(2, explicit=True):
                        accum[acc_id] = V.vadd(accum[acc_id], accum[acc_id + 2])
                    accum[0] = V.vadd(accum[0], accum[1])

                    mask1 = V.create_mask(1, size=1)
                    alpha1 = V.vbrc(T.float32(alpha), size=1)
                    norm = V.vsqrt(V.vcadd(accum[0], mask, reassoc=True))
                    output_scale = V.vdiv(alpha1, V.vadds(norm, T.float32(eps), mask1), mask1, precision='exact')
                    output_scale_broadcast = V.vbrc(output_scale, size=vec_size)
                    S.mem_bar('VST_VLD')
                    for chunk in T.serial(num_chunks):
                        offset = chunk * vec_size
                        V.vstore(V.vmul(V.vload(out_ub[offset], size=vec_size), output_scale_broadcast), out_ub[offset])

                T.copy(out_ub, out[row, :])

    return engram_sinkhorn_finalize_kernel_vmi
