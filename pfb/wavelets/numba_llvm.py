import numba
from numba.core import types, cgutils, typing
from numba.extending import intrinsic
from numba.np.arrayobj import make_view, _change_dtype
from numba.core.imputils import impl_ret_borrowed


@intrinsic
def force_type_contiguity(typingctx, array):
    return_type = array.copy(layout="C")
    sig = return_type(array)

    def codegen(context, builder, signature, args):
        def check(array):
            if not array.flags.c_contiguous:
                raise ValueError("Attempted to force type contiguity "
                                 "on an array whose layout is "
                                 "non-contiguous")

        check_sig = typing.signature(types.none, signature.args[0])
        context.compile_internal(builder, check, check_sig, args)

        return impl_ret_borrowed(context, builder, return_type, args[0])

    return sig, codegen


@intrinsic
def slice_axis(typingctx, array, index, axis):
    # Return a single array, not necessarily contiguous
    return_type = array.copy(ndim=1, layout="A")

    if not isinstance(array, types.Array):
        raise TypeError("array is not an Array")

    if "C" not in array.layout:
        raise TypeError("array must be C contiguous")

    if (not isinstance(index, types.UniTuple) or
        not isinstance(index.dtype, types.Integer)):
        raise TypeError("index is not a Homogenous Tuple of Integers")

    if len(index) != array.ndim:
        raise TypeError("array.ndim != len(index")

    if not isinstance(axis, types.Integer):
        raise TypeError("axis is not an Integer")

    sig = return_type(array, index, axis)

    def codegen(context, builder, signature, args):
        array_type, idx_type, axis_type = signature.args
        array, idx, axis = args
        array = context.make_array(array_type)(context, builder, array)

        zero = context.get_constant(types.intp, 0)
        llvm_intp_t = context.get_value_type(types.intp)
        ndim = array_type.ndim

        view_shape = cgutils.alloca_once(builder, llvm_intp_t)
        view_stride = cgutils.alloca_once(builder, llvm_intp_t)

        # Final array indexes. We only know the slicing index at runtime
        # so we need to recreate idx but with zero at the slicing axis
        indices = cgutils.alloca_once(builder, llvm_intp_t, size=array_type.ndim)

        for ax in range(array_type.ndim):
            llvm_ax = context.get_constant(types.intp, ax)
            predicate = builder.icmp_unsigned("!=", llvm_ax, axis)

            with builder.if_else(predicate) as (not_equal, equal):
                with not_equal:
                    # If this is not the slicing axis,
                    # use the appropriate tuple index
                    value = builder.extract_value(idx, ax)
                    builder.store(value, builder.gep(indices, [llvm_ax]))

                with equal:
                    # If this is the slicing axis,
                    # store zero as the index.
                    # Also record the stride and shape
                    builder.store(zero, builder.gep(indices, [llvm_ax]))
                    size = builder.extract_value(array.shape, ax)
                    stride = builder.extract_value(array.strides, ax)
                    builder.store(size, view_shape)
                    builder.store(stride, view_stride)

        # Build a python list from indices
        tmp_indices = []

        for i in range(ndim):
            i = context.get_constant(types.intp, i)
            tmp_indices.append(builder.load(builder.gep(indices, [i])))

        # Get the data pointer obtained from indexing the array
        dataptr = cgutils.get_item_pointer(context, builder,
                                           array_type, array,
                                           tmp_indices,
                                           wraparound=True,
                                           boundscheck=True)

        # Set up the shape and stride. There'll only be one
        # dimension, corresponding to the axis along which we slice
        view_shapes = [builder.load(view_shape)]
        view_strides = [builder.load(view_stride)]

        # Make a view with the data pointer, shapes and strides
        retary = make_view(context, builder,
                           array_type, array, return_type,
                           dataptr, view_shapes, view_strides)

        result = retary._getvalue()
        return impl_ret_borrowed(context, builder, return_type, result)

    return sig, codegen