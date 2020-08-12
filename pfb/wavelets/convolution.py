import numba

from pfb.wavelets.modes import Modes

@numba.njit(nogil=True)
def downsampling_convolution(input, output, filter, mode, step):
    i = step - 1
    o = 0
    N = input.shape[0]
    F = filter.shape[0]

    if mode is Modes.smooth and N < 2:
        mode = Modes.constant_edge

    # left boundary overhang
    while i < F and i < N:
        fsum = input.dtype.type(0)
        j = 0

        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1

        if mode is Modes.symmetric:
            k = 0

            while k < N and j < F:
                fsum += filter[j] * input[k]
                j += 1
                k += 1

            k = 0

            while k < N and j < F:
                fsum += filter[j] * input[N - 1 - k]
                j += 1
                k += 1

        elif mode is Modes.asymmetric:
            raise NotImplementedError("asymmetric convolution not implemented")
        elif mode is Modes.reflect:
            raise NotImplementedError("reflect convolution not implemented")
        elif mode is Modes.antireflect:
            raise NotImplementedError("antireflect convolution not implemented")
        elif mode is Modes.constant_edge:
            raise NotImplementedError("constant_edge convolution not implemented")
        elif mode is Modes.smooth:
            raise NotImplementedError("smooth convolution not implemented")
        elif mode is Modes.periodic:
            raise NotImplementedError("periodic convolution not implemented")
        elif mode is Modes.zeropad:
            pass
        else:
            raise ValueError("Unknown mode")

        output[o] = fsum

        i += step
        o += 1

    # center (if input equal or wider than filter: N >= F)
    while i < N:
        fsum = input.dtype.type(0)
        j = 0

        while j < F:
            fsum += input[i - j] * filter[j]
            j += 1

        output[o] = fsum

        i += step
        o += 1

    # center (if filter is wider than input: F > N)
    while i < F:
        fsum = input.dtype.type(0)
        j = 0

        if mode is Modes.symmetric:
            while i - j >= N:
                k = 0

                while k < N and i - j >= N:
                    fsum += filter[i - N - j] * input[N - 1 - k]
                    j += 1
                    k += 1

                k = 0

                while k < N and i - j >= N:
                    fsum += filter[i - N - j] * input[k]
                    j += 1
                    k += 1

        elif mode is Modes.asymmetric:
            raise NotImplementedError("asymmetric convolution not implemented")
        elif mode is Modes.reflect:
            raise NotImplementedError("reflect convolution not implemented")
        elif mode is Modes.antireflect:
            raise NotImplementedError("antireflect convolution not implemented")
        elif mode is Modes.constant_edge:
            raise NotImplementedError("constant_edge convolution not implemented")
        elif mode is Modes.smooth:
            raise NotImplementedError("smooth convolution not implemented")
        elif mode is Modes.periodic:
            raise NotImplementedError("periodic convolution not implemented")
        elif mode is Modes.zeropad:
            j = i - N + 1
        else:
            raise ValueError("Unknown mode")


        while j <= i:
            fsum += filter[j] * input[i - j]
            j += 1


        if mode is Modes.symmetric:
            while j < F:
                k = 0

                while k < N and j < F:
                    fsum += filter[j] * input[k]
                    j += 1
                    k += 1

                k = 0

                while k < N and j < F:
                    fsum += filter[j] * input[N - 1 - k]
                    j += 1
                    k += 1

        elif mode is Modes.asymmetric:
            raise NotImplementedError("asymmetric convolution not implemented")
        elif mode is Modes.reflect:
            raise NotImplementedError("reflect convolution not implemented")
        elif mode is Modes.antireflect:
            raise NotImplementedError("antireflect convolution not implemented")
        elif mode is Modes.constant_edge:
            raise NotImplementedError("constant_edge convolution not implemented")
        elif mode is Modes.smooth:
            raise NotImplementedError("smooth convolution not implemented")
        elif mode is Modes.periodic:
            raise NotImplementedError("periodic convolution not implemented")
        elif mode is Modes.zeropad:
            pass
        else:
            raise ValueError("Unknown mode")

        output[o] = fsum

        i += step
        o += 1

    # right boundary overhang
    while i < N + F - 1:
        fsum = input.dtype.type(0)
        j = 0

        if mode is Modes.symmetric:
            while i - j >= N:
                k = 0

                while k < N and i - j >= N:
                    fsum += filter[i - N - j] * input[N - 1 - k]
                    j += 1
                    k += 1

                k = 0

                while k < N and i - j >= N:
                    fsum += filter[i - N - j] * input[k]
                    j += 1
                    k += 1

        elif mode is Modes.asymmetric:
            raise NotImplementedError("asymmetric convolution not implemented")
        elif mode is Modes.reflect:
            raise NotImplementedError("reflect convolution not implemented")
        elif mode is Modes.antireflect:
            raise NotImplementedError("antireflect convolution not implemented")
        elif mode is Modes.constant_edge:
            raise NotImplementedError("constant_edge convolution not implemented")
        elif mode is Modes.smooth:
            raise NotImplementedError("smooth convolution not implemented")
        elif mode is Modes.periodic:
            raise NotImplementedError("periodic convolution not implemented")
        elif mode is Modes.zeropad:
            j = i - N + 1
        else:
            raise ValueError("Unknown mode")


        while j < F:
            fsum += filter[j] * input[i - j]
            j += 1

        output[o] = fsum

        i += step
        o += 1

    if o != output.shape[0]:
        raise ValueError("Output mismatch")
