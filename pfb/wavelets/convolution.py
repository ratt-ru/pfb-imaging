import numba

from pfb.wavelets.modes import Modes


@numba.njit(nogil=True, cache=True)
def upsampling_convolution_valid_sf(input, filter, output, mode):
    if mode is Modes.periodisation:
        raise NotImplementedError("periodisation upsampling not implemented")

    N = input.shape[0]
    F = filter.shape[0]
    O = output.shape[0]

    if F % 2:
        raise ValueError("even filter required for upsampling")

    if N < (F // 2):
        raise ValueError("input.shape[0] < (filter.shape[0] // 2)")

    o = 0
    i = (F // 2) - 1

    while i < N:
        sum_even = input.dtype.type(0)
        sum_odd = input.dtype.type(0)
        j = 0

        while j < F // 2:
            sum_even += filter[j*2] * input[i-j]
            sum_odd += filter[j*2+1] * input[i-j]

            j += 1

        output[o] += sum_even
        output[o+1] += sum_odd

        i += 1
        o += 2


@numba.njit(nogil=True, cache=True)
def downsampling_convolution(input, output, filter, mode, step):
    if mode is Modes.periodisation:
        raise NotImplementedError("periodisation downsampling not implemented")

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
            raise NotImplementedError("asymmetric downsampling not implemented")
        elif mode is Modes.reflect:
            raise NotImplementedError("reflect downsampling not implemented")
        elif mode is Modes.antireflect:
            raise NotImplementedError("antireflect downsampling not implemented")
        elif mode is Modes.constant_edge:
            raise NotImplementedError("constant_edge downsampling not implemented")
        elif mode is Modes.smooth:
            raise NotImplementedError("smooth downsampling not implemented")
        elif mode is Modes.periodic:
            raise NotImplementedError("periodic downsampling not implemented")
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
            raise NotImplementedError("asymmetric downsampling not implemented")
        elif mode is Modes.reflect:
            raise NotImplementedError("reflect downsampling not implemented")
        elif mode is Modes.antireflect:
            raise NotImplementedError("antireflect downsampling not implemented")
        elif mode is Modes.constant_edge:
            raise NotImplementedError("constant_edge downsampling not implemented")
        elif mode is Modes.smooth:
            raise NotImplementedError("smooth downsampling not implemented")
        elif mode is Modes.periodic:
            raise NotImplementedError("periodic downsampling not implemented")
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
            raise NotImplementedError("asymmetric downsampling not implemented")
        elif mode is Modes.reflect:
            raise NotImplementedError("reflect downsampling not implemented")
        elif mode is Modes.antireflect:
            raise NotImplementedError("antireflect downsampling not implemented")
        elif mode is Modes.constant_edge:
            raise NotImplementedError("constant_edge downsampling not implemented")
        elif mode is Modes.smooth:
            raise NotImplementedError("smooth downsampling not implemented")
        elif mode is Modes.periodic:
            raise NotImplementedError("periodic downsampling not implemented")
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
            raise NotImplementedError("asymmetric downsampling not implemented")
        elif mode is Modes.reflect:
            raise NotImplementedError("reflect downsampling not implemented")
        elif mode is Modes.antireflect:
            raise NotImplementedError("antireflect downsampling not implemented")
        elif mode is Modes.constant_edge:
            raise NotImplementedError("constant_edge downsampling not implemented")
        elif mode is Modes.smooth:
            raise NotImplementedError("smooth downsampling not implemented")
        elif mode is Modes.periodic:
            raise NotImplementedError("periodic downsampling not implemented")
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
