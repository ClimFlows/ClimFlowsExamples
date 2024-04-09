function kernel_100fma(a, b, c, out)
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    @inbounds if i <= length(out)
        aa, bb, cc = a[i], b[i], c[i]
        for j in 1:33
            aa = muladd(aa, bb, cc)
            bb = muladd(aa, bb, cc)
            cc = muladd(aa, bb, cc)
        end
        out[i] = muladd(aa, bb, cc)
    end
    return
end

function CUDA_flops(n::Integer=5000, dev::CuDevice=CuDevice(0))
    device!(dev) do
        dims = (n, n)
        len = prod(dims)

        a, b, c = ( round.(rand(Float32, dims) * 100) for i=1:3)
        out = similar(a)
        d_a, d_b, d_c, d_out = map(CuArray, (a,b,c,out))

        kernel = @cuda launch=false kernel_100fma(d_a, d_b, d_c, d_out)
        config = launch_configuration(kernel.fun)
        threads = min(len, config.threads)
        blocks = cld(len, threads)
        flopcount = 200*len

        # warm-up
        kernel(d_a, d_b, d_c, d_out ; threads, blocks)
        CUDA.synchronize()
        return flopcount / CUDA.@elapsed kernel(d_a, d_b, d_c, d_out; threads, blocks)
    end
end

CUDA.versioninfo()
flops = CUDA_flops()
@info "Peak flops: $(flops*1e-12) TFlops"
