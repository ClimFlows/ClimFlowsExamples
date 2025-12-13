using Pkg; Pkg.activate(@__DIR__)
using Revise

using FastGaussQuadrature
using Polynomials

except(N,i) = filter(j->(j != i), 1:N)

function ddx(gauss, lobatto, i, j)
    N = length(gauss)
    roots = lobatto[except(N+1,i)]
    p = fromroots(FactoredPolynomial, roots)
    p = p/p(lobatto[i])
    dp = derivative(p)
    return dp(gauss[j])
end

function ddx(N)
    gauss, wg = gausslegendre(N)
    lobatto, wl = gausslobatto(N+1)
    D = [ddx(gauss, lobatto, i, j) for i=1:N+1, j=1:N]
    return gauss, wg, lobatto, wl, D
end

M, N = 3, 3
gauss, wg, lobatto, wl, D = ddx(N)
@info "derivative" gauss lobatto D (-D*gauss)' lobatto'*D

DD=zero(randn(M*N,M*N+1))
x = eltype(DD)[]
wx = eltype(DD)[]
y = [lobatto[1]]
wy = zero(randn(M*N+1))
for k=0:(M-1)
    for j=1:N
        DD[k*N+j, k*N+1:(k+1)*N+1] = D[:,j]
        push!(x, gauss[j]+2k)
        push!(wx, wg[j])
        push!(y, lobatto[j+1]+2k)
    end
    for j=1:N+1
        wy[k*N+j] += wl[j]
    end
end

# x = [gauss[1] ; gauss[2] ; gauss[1]+2 ; gauss[2]+2]
# y = [lobatto[1] ; lobatto[2] ; lobatto[3] ; lobatto[2]+2 ; lobatto[3]+2]
# ww = [wl[1] ; wl[2]; wl[1]+wl[3] ; wl[2]; wl[3]]
df_y(DD, wx, wy, x) = -DD'*(wx.*x)./wy
df_y(DD, wx, wy, x.^N) - N*y.^(N-1)
DD*(y.^(N+1))-(N+1)*x.^N

# P4/D3
#   |A a B b C c D|D d E z F f G|
# f(A,B,C,D) => f'(a,b,c)
# f(D,E,F,G) => f'(d,e,f)
#  f'(a)    | Aa Ba Ca Da          |   | f(A)
#  f'(b)    | Ab Bb Cb Db          |   | f(B)
#  f'(c)    | Ac Bc Cc Dc          |   | f(C)
#  f'(d) =  |          Aa Ba Ca Da | × | f(D)
#  f'(e)    |          Ab Bb Cb Db |   | f(E)
#  f'(f)    |          Ac Bc Cc Dc |   | f(F)
#                                      | f(G)
# Unknowns: N^2+3N-1 = 17 
#    2N-1=5 quadrature points (a,b,c, B, C)
#    N*(N+1) = 12 coefficients
# Orders p,q => N*p + (N+1)*q constraints
#    
# N=2M+1 N^2+3N-1 = N(2M+1)+3(2M+1)-1 = 2MN + 8M + 3 
#    p,q=M+1,M => N*p + (N+1)*q = N*(M+1)+(N+1)*M = 2MN+M+N   
#    p,q=M+1,M+1 => N*p + (N+1)*q = N*(M+1)+(N+1)*(M+1) = 2MN+5M + 3   
#    p,q=M+2,M+1 => N*p + (N+1)*q = N*(M+2)+(N+1)*(M+1) = 2MN+7M + 4
