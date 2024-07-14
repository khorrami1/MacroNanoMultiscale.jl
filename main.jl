
using Ferrite
using FerriteAssembly
using ProgressMeter

include("initialize.jl")

# FEM implementation, using Ferrite.jl package
# using Ferrite
# using FerriteAssembly

# Generate a grid
N = 2
L = 100.0
#3D
left = zero(Vec{3})
right = L * ones(Vec{3})
grid = generate_grid(Tetrahedron, (N, N, 1), left, right)

#2D
# left = zero(Vec{2})
# right = L * ones(Vec{2})
# grid = generate_grid(Triangle, (N, N), left, right)

# Finite element base
#3D
ip = Lagrange{RefTetrahedron, 1}()^3
qr = QuadratureRule{RefTetrahedron}(1)
qr_face = FacetQuadratureRule{RefTetrahedron}(1)

#2D
# ip = Lagrange{RefTriangle, 1}()^2
# qr = QuadratureRule{RefTriangle}(1)
# qr_face = FacetQuadratureRule{RefTriangle}(1)

cv = CellValues(qr, ip)
fv = FacetValues(qr_face, ip)
# DofHandler
dh = DofHandler(grid)
add!(dh, :u, ip) # Add a displacement field
close!(dh)

timestep = 5e-3
strain_rate = 5e-3

arve_ref = get_ARVE_from_lmp(lmp)

ARVE_State = Union{Tensor{2,3}, ARVE}

function FerriteAssembly.create_cell_state(::Nothing, cv::CellValues, args...)
    cell_state_qp = Vector{ARVE_State}(undef, 2)
    cell_state_qp[1] = one(Tensor{2,3})
    cell_state_qp[2] = arve_ref
    return [cell_state_qp for _ in 1:getnquadpoints(cv)]
end

# function FerriteAssembly.element_residual!(re, state, ue, ::Nothing, cv::CellValues, buffer)
#     #Δt = FerriteAssembly.get_time_increment(buffer)
#     old_states = FerriteAssembly.get_old_state(buffer)
#     ndofs = getnbasefunctions(cv)
#     # lmp = buffer.user_cache
#     for qp in 1:getnquadpoints(cv)
#         old_state = old_states[qp]
#         dΩ = getdetJdV(cv, qp)
#         ∇u = function_gradient(cv, qp, ue)
#         F = one(∇u) + ∇u # F is a Tensor
#         # set_ARVE_to_lmp!(lmp, old_state[2])
#         state[qp][2] = apply_C_ARVE!(lmp, old_state[2], F, timestep, strain_rate)
#         # σ_voigt, C_voigt = calc_S_C(lmp)
#         σ_voigt = get_stress_gpa(lmp)
#         σ = fromvoigt(SymmetricTensor{2,3}, σ_voigt)
#         P = det(F) * σ ⋅ transpose(inv(F))
#         # Calculation ∂P∂F
#         # C = fromvoigt(SymmetricTensor{4,3}, C_voigt)
#         # Loop over test function
#         for i in 1:ndofs
#             δϵ = shape_symmetric_gradient(cv, qp, i)
#             re[i] += (δϵ ⊡ P) * dΩ
#             # ∇δui∂P∂F = ∇δui ⊡ ∂P∂F
#             # for j in 1:ndofs
#             #     ∇δuj = shape_gradient(cv, qp, j)
#             #     ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
#             # end
#         end
#     end
# end

function FerriteAssembly.element_routine!(ke, re, state, ue, ::Nothing, cv::CellValues, buffer)
    #Δt = FerriteAssembly.get_time_increment(buffer)
    old_states = FerriteAssembly.get_old_state(buffer)
    ndofs = getnbasefunctions(cv)
    # lmp = buffer.user_cache
    for qp in 1:getnquadpoints(cv)
        old_state = old_states[qp]
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u # F is a Tensor
        # set_ARVE_to_lmp!(lmp, old_state[2])
        state[qp][2] = apply_C_ARVE!(lmp, old_state[2], F, timestep, strain_rate)
        σ_voigt, C_voigt = calc_S_C(lmp)
        # σ_voigt = get_stress_gpa(lmp_ref)
        σ = fromvoigt(SymmetricTensor{2,3}, σ_voigt)
        P = det(F) * σ ⋅ transpose(inv(F))
        # Calculation ∂P∂F
        C = fromvoigt(SymmetricTensor{4,3}, C_voigt)
        # Loop over test function
        for i in 1:ndofs
            δϵ = shape_symmetric_gradient(cv, qp, i)
            re[i] += (δϵ ⊡ P) * dΩ
            # ∇δui∂P∂F = ∇δui ⊡ ∂P∂F
            for j in 1:ndofs
                Δϵ = shape_symmetric_gradient(cv, qp, j)
                ke[i, j] += (Δϵ ⊡ C ⊡ δϵ) * dΩ
            end
        end
    end
end

r = zeros(ndofs(dh))
u = zeros(ndofs(dh))
# reAssembler = ReAssembler(r)

# it must be before setup_domainbuffer
# FerriteAssembly.allocate_cell_cache(::Any, ::Any) = lmp_ref
# threading=true causes an error! because of multi-threading in LAMMPS
buffer = setup_domainbuffer(DomainSpec(dh, nothing, cv); threading=false)


# work!(reAssembler, buffer; a=u)

K = allocate_matrix(dh)
# assembler = start_assemble(K, r)
 
ch = ConstraintHandler(dh)

dbc1 = Dirichlet(:u, getfacetset(grid, "left" ), (x,t)->[0.1*t,0.,0.], [1,2,3])
dbc2 = Dirichlet(:u, getfacetset(grid, "right"), (x,t)->[0.,0.], [1,2])

add!(ch, dbc1)
add!(ch, dbc2)
close!(ch)

Ferrite.update!(ch, 0.)

f = zeros(ndofs(dh))
apply!(K, f, ch)
# apply!(u, ch)
freeDofs = ch.free_dofs
preDofs = ch.prescribed_dofs

# u[freeDofs] .= -K[freeDofs, freeDofs]\r[freeDofs] # Singularity

# u[freeDofs] .= -K[preDofs, freeDofs]\r[preDofs]

du = zeros(ndofs(dh))

itr = 0
# @run work!(assembler, buffer; a=u)
# @show norm(r[freeDofs])

u_old = zeros(ndofs(dh))
# t_old = 0

maxIter = 10
tolerance = 0.1

@showprogress for t in 1:10
    update!(ch, t)
    apply!(u, ch)
    # Update and apply the Neumann boundary conditions
    # fill!(f, 0)
    # apply!(f, lh, t)
    # set_time_increment!(buffer, t-t_old)
    du .= u .- u_old
    for i in 1:maxIter
        # Assemble the system
        assembler = start_assemble(K, r)
        work!(assembler, buffer; a=du)
        # r .-= f
        # Apply boundary conditions
        apply_zero!(K, r, ch)
        # Check convergence
        norm_r = norm(r)
        println("")
        @show t, i, norm_r
        if (norm_r < 100*tolerance) 
            break;
        elseif (i == maxIter)
            @warn "Did not converge"
        end
        # Solve the linear system and update the dof vector
        du .+= K\r
        # @show du
    end
    u .+= du
    u_old .= u
    apply!(u, ch) # Make sure Dirichlet BC are exactly fulfilled
    update_states!(buffer)
    update_grid!(dh, du)
end
