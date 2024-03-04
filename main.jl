
include("initialize.jl")

# FEM implementation, using Ferrite.jl package
# using Ferrite
# using FerriteAssembly

# Generate a grid
N = 2
L = 10.0
left = zero(Vec{3})
right = L * ones(Vec{3})
grid = generate_grid(Tetrahedron, (N, N, 1), left, right)

# Finite element base
ip = Lagrange{RefTetrahedron, 1}()^3
qr = QuadratureRule{3, RefTetrahedron}(1)
qr_face = QuadratureRule{2, RefTetrahedron}(1)
cv = CellValues(qr, ip)
fv = FaceValues(qr_face, ip)
# DofHandler
dh = DofHandler(grid)
add!(dh, :u, 3) # Add a displacement field
close!(dh)

timestep = 5e-3
strain_rate = 1e-3

# Base.@kwdef struct ARVEmat{T} 

# end


arve_ref = get_ARVE_from_lmp(lmp_ref)

ARVE_State = Union{Tensor{2,3}, ARVE}

function FerriteAssembly.create_cell_state(::Nothing, cv::CellValues, args...)
    cell_state_qp = Vector{ARVE_State}(undef, 2)
    cell_state_qp[1] = one(Tensor{2,3})
    cell_state_qp[2] = arve_ref
    return [cell_state_qp for _ in 1:getnquadpoints(cv)]
end


function FerriteAssembly.element_residual!(re, state, ue, ::Nothing, cv::CellValues, buffer)
    #Δt = FerriteAssembly.get_time_increment(buffer)
    old_states = FerriteAssembly.get_old_state(buffer)
    ndofs = getnbasefunctions(cv)
    lmp = buffer.user_cache
    for qp in 1:getnquadpoints(cv)
        old_state = old_states[qp]
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u # F is a Tensor
        set_ARVE_to_lmp!(lmp, old_state[2])
        state[qp][2] = apply_C_ARVE!(lmp, old_state[2], F, timestep, strain_rate)
        # σ_voigt, C_voigt = calc_S_C(lmp)
        σ_voigt = get_stress_gpa(lmp)
        σ = fromvoigt(SymmetricTensor{2,3}, σ_voigt)
        # P = det(F) * σ ⋅ transpose(inv(F))
        # Calculation ∂P∂F
        # C = fromvoigt(SymmetricTensor{4,3}, C_voigt)
        # Loop over test function
        for i in 1:ndofs
            δϵ = shape_symmetric_gradient(cv, qp, i)
            re[i] += (δϵ ⊡ σ) * dΩ
            # ∇δui∂P∂F = ∇δui ⊡ ∂P∂F
            # for j in 1:ndofs
            #     ∇δuj = shape_gradient(cv, qp, j)
            #     ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            # end
        end
    end
end

function FerriteAssembly.element_routine!(ke, re, state, ue, ::Nothing, cv::CellValues, buffer)
    #Δt = FerriteAssembly.get_time_increment(buffer)
    old_states = FerriteAssembly.get_old_state(buffer)
    ndofs = getnbasefunctions(cv)
    lmp = buffer.user_cache
    for qp in 1:getnquadpoints(cv)
        old_state = old_states[qp]
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u # F is a Tensor
        set_ARVE_to_lmp!(lmp, old_state[2])
        state[qp][2] = apply_C_ARVE!(lmp, old_state[2], F, timestep, strain_rate)
        σ_voigt, C_voigt = calc_S_C(lmp)
        # σ_voigt = get_stress_gpa(lmp_ref)
        σ = fromvoigt(SymmetricTensor{2,3}, σ_voigt)
        # P = det(F) * σ ⋅ transpose(inv(F))
        # Calculation ∂P∂F
        C = fromvoigt(SymmetricTensor{4,3}, C_voigt)
        # Loop over test function
        for i in 1:ndofs
            δϵ = shape_symmetric_gradient(cv, qp, i)
            re[i] += (δϵ ⊡ σ) * dΩ
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
FerriteAssembly.allocate_cell_cache(::Any, ::Any) = lmp_ref
# threading=true causes an error! because of multi-threading in LAMMPS
buffer = setup_domainbuffer(DomainSpec(dh, nothing, cv); threading=false)


# work!(reAssembler, buffer; a=u)

K = create_sparsity_pattern(dh)
assembler = start_assemble(K, r)
 
ch = ConstraintHandler(dh)

dbc1 = Dirichlet(:u, getfaceset(grid, "left" ), (x,t)->[0.,0.,0.], [1,2,3])
dbc2 = Dirichlet(:u, getfaceset(grid, "right"), (x,t)->[0.,0.,0.], [1,2,3])

add!(ch, dbc1)
add!(ch, dbc2)
close!(ch)

Ferrite.update!(ch, 0.)

# f = zeros(ndofs(dh))
# apply!(K, f, ch)
# apply!(u, ch)
freeDofs = ch.free_dofs
preDofs = ch.prescribed_dofs

# u[freeDofs] .= -K[freeDofs, freeDofs]\r[freeDofs] # Singularity

# u[freeDofs] .= -K[preDofs, freeDofs]\r[preDofs]

du = zeros(ndofs(dh))

itr = 0
# work!(assembler, buffer; a=u)
# @show norm(r[freeDofs])

u0 = zeros(ndofs(dh))

while true 
    @show itr += 1

    if itr > 10
        break
    end
    
    # du[freeDofs] .= K[preDofs, freeDofs]\r[preDofs]
    # u[freeDofs] .+= du[freeDofs]

    work!(assembler, buffer; a=du)
    update_states!(buffer)

    @show norm_r = norm(r[freeDofs])

    if norm_r < 1e-3
        break
    end

    apply_zero!(K, r, ch)
    du = Symmetric(K) \ r
    u -= du
    apply!(u, ch)

end