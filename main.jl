using LAMMPS
using LinearAlgebra

lmp_ref = LMP(["-pk","omp", "12", "-sf", "omp"])


test_str = "
clear 
units metal 
dimension 3
boundary p p p 
atom_style 	atomic
atom_modify map array sort 0 0
box tilt large
read_data ARVE_10_relaxed.data
#read_restart ARVE_10_relaxed.equil
pair_style eam/alloy
pair_coeff * * CuAgAu_Zhou04.eam.alloy Au
thermo_style custom step pxx pyy pzz pxy pxz pyz
timestep 0.005
thermo 1000
variable Temp equal 1
#min_style cg
#fix fix_boxRelax all box/relax aniso 0.0
#minimize  1.0e-10 1.0e-10 2000 2000
#write_restart L=80,DR=1_equilibrium.equil
#unfix fix_boxRelax
fix fix_nvt all nvt temp \${Temp} \${Temp} 1
run 100
"

#str_restart = "read_restart L=80,DR=1_equilibrium.equil"
#command(lmp_ref, str_restart)

LAMMPS.API.lammps_commands_string(lmp_ref, test_str)

command(lmp_ref, "run 1000")

# you can see the atoms in Ovito
# command(lmp_ref, "write_dump all atom ARVE_10_relaxed.atom")
# command(lmp_ref, "write_data ARVE_10_relaxed.data")

function get_stress(lmp)
    Pxx = LAMMPS.API.lammps_get_thermo(lmp, "pxx")
    Pyy = LAMMPS.API.lammps_get_thermo(lmp, "pyy")
    Pzz = LAMMPS.API.lammps_get_thermo(lmp, "pzz")
    Pxy = LAMMPS.API.lammps_get_thermo(lmp, "pxy")
    Pxz = LAMMPS.API.lammps_get_thermo(lmp, "pxz")
    Pyz = LAMMPS.API.lammps_get_thermo(lmp, "pyz")
    return [-Pxx, -Pyy, -Pzz, -Pxy, -Pxz, -Pyz]
end

get_stress(lmp_ref)*1e-4 # units in GPa

struct ARVE_Box
    xlo :: Float64
    xhi :: Float64
    ylo :: Float64
    yhi :: Float64
    zlo :: Float64
    zhi :: Float64
    xy :: Float64
    xz :: Float64
    yz :: Float64
end

struct ARVE
    x :: Matrix{Float64} # coordinates of atoms
    v :: Matrix{Float64} # velocities of atoms
    box :: ARVE_Box # LAMMPS box of ARVE
end


function get_box(lmp::LMP)
    xlo = LAMMPS.API.lammps_get_thermo(lmp, "xlo")
    xhi = LAMMPS.API.lammps_get_thermo(lmp, "xhi")
    ylo = LAMMPS.API.lammps_get_thermo(lmp, "ylo")
    yhi = LAMMPS.API.lammps_get_thermo(lmp, "yhi")
    zlo = LAMMPS.API.lammps_get_thermo(lmp, "zlo")
    zhi = LAMMPS.API.lammps_get_thermo(lmp, "zhi")
    xy = LAMMPS.API.lammps_get_thermo(lmp, "xy")
    xz = LAMMPS.API.lammps_get_thermo(lmp, "xz")
    yz = LAMMPS.API.lammps_get_thermo(lmp, "yz")
    return  ARVE_Box(xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz)
end


function get_ARVE_from_lmp(lmp::LMP)
    x = extract_atom(lmp, "x")
    v = extract_atom(lmp, "v")
    rve_box = get_box(lmp)
    return ARVE(x, v, rve_box)
end

arve_old = deepcopy(get_ARVE_from_lmp(lmp_ref))
arve_new = deepcopy(arve_old)

function set_ARVE_to_lmp!(lmp::LMP, arve::ARVE)
    command(lmp, "change_box all x final "*string(arve.box.xlo)*" "*
    string(arve.box.xhi)*" y final "*string(arve.box.ylo)*" "*string(arve.box.yhi)*
    " z final "*string(arve.box.zlo)*" "*string(arve.box.zhi)*" xy final "*string(arve.box.xy)*
    " xz final "*string(arve.box.xz)*" yz final "*string(arve.box.yz))
    LAMMPS.API.lammps_scatter_atoms(lmp, "x", 1, 3, arve.x)
    LAMMPS.API.lammps_scatter_atoms(lmp, "v", 1, 3, arve.v)
end

set_ARVE_to_lmp!(lmp_ref, arve_new)

function apply_C_ARVE!(lmp::LMP, arve::ARVE, F::Tensor{2,3}, timestep, strain_rate)
    C = transpose(F)⋅F
    detF = sqrt(det(C))
    invC = inv(C)

    set_ARVE_to_lmp!(lmp, arve)

    xlo = LAMMPS.API.lammps_get_thermo(lmp, "xlo")
    ylo = LAMMPS.API.lammps_get_thermo(lmp, "ylo")
    zlo = LAMMPS.API.lammps_get_thermo(lmp, "zlo")
    xhi = LAMMPS.API.lammps_get_thermo(lmp, "xhi")
    yhi = LAMMPS.API.lammps_get_thermo(lmp, "yhi")
    zhi = LAMMPS.API.lammps_get_thermo(lmp, "zhi")
    xy = LAMMPS.API.lammps_get_thermo(lmp, "xy")
    xz = LAMMPS.API.lammps_get_thermo(lmp, "xz")
    yz = LAMMPS.API.lammps_get_thermo(lmp, "yz")
    lx = LAMMPS.API.lammps_get_thermo(lmp, "lx")
    ly = LAMMPS.API.lammps_get_thermo(lmp, "ly")
    lz = LAMMPS.API.lammps_get_thermo(lmp, "lz")
    a = lx*Tensor{1,3}((1.0,0.0,0.0))
    b = xy*Tensor{1,3}((1.0,0.0,0.0)) + ly*Tensor{1,3}((0.0,1.0,0.0))
    c = xz*Tensor{1,3}((1.0,0.0,0.0)) + yz*Tensor{1,3}((0.0,1.0,0.0)) + lz*Tensor{1,3}((0.0,0.0,1.0))
    corss_a_b = cross(a,b)
    dot_a_Ca = dot(a, C⋅a)
    temp_var = sqrt(dot(corss_a_b, invC ⋅ corss_a_b))
    lx_new = sqrt(dot_a_Ca)
    ly_new = detF * temp_var/sqrt(dot_a_Ca)
    lz_new = dot(corss_a_b, c)/temp_var
    xy_new =  dot(a, C⋅b)/sqrt(dot_a_Ca)
    xz_new =  dot(a, C⋅c)/sqrt(dot_a_Ca)
    yz_new =  dot(corss_a_b, invC ⋅ cross(a,c))/(temp_var*sqrt(dot_a_Ca))
    xhi_new = xlo + lx_new
    yhi_new = ylo + ly_new
    zhi_new = zlo + lz_new
    command(lmp, "fix fix_deform all deform 1 x final "*string(xlo)*" "*
        string(xhi_new)*" y final "*string(ylo)*" "*string(yhi_new)*
        " z final "*string(zlo)*" "*string(zhi_new)*" xy final "*string(xy_new)*
        " xz final "*string(xz_new)*" yz final "*string(yz_new)*" remap x units box")
    #delta = [lx_new - lx, ly_new - ly, lz_new - lz, xy_new - xy, xz_new - xz, yz_new - yz]
    # num_MD_steps = Int(ceil(maximum(delta) / (epsilon_rate * timestep)))
    num_MD_steps = Int(ceil(maximum(abs.(F - one(Tensor{2,3}))) / (strain_rate * timestep)))
    command(lmp, "run "*string(num_MD_steps))

    return deepcopy(get_ARVE_from_lmp(lmp))
end

function calc_S_C(lmp; eps=1e-5)

    S = zeros(6)
    C = zeros(6, 6)

    S[1] = LAMMPS.API.lammps_get_thermo(lmp, "pxx") * 1e-4
    S[2] = LAMMPS.API.lammps_get_thermo(lmp, "pyy") * 1e-4
    S[3] = LAMMPS.API.lammps_get_thermo(lmp, "pzz") * 1e-4
    S[4] = LAMMPS.API.lammps_get_thermo(lmp, "pyz") * 1e-4
    S[5] = LAMMPS.API.lammps_get_thermo(lmp, "pxz") * 1e-4
    S[6] = LAMMPS.API.lammps_get_thermo(lmp, "pxy") * 1e-4

    xyEps = LAMMPS.API.lammps_get_thermo(lmp, "xy")*eps
    xzEps = LAMMPS.API.lammps_get_thermo(lmp, "xz")*eps
    yzEps = LAMMPS.API.lammps_get_thermo(lmp, "yz")*eps
    lxEps = LAMMPS.API.lammps_get_thermo(lmp, "lx")*eps
    lyEps = LAMMPS.API.lammps_get_thermo(lmp, "ly")*eps
    lzEps = LAMMPS.API.lammps_get_thermo(lmp, "lz")*eps

    # voigt_str = ["x", "y", "z", "yz", "xz", "xy"]
    voigt_str_p = ["pxx", "pyy", "pzz", "pyz", "pxz", "pxy"]

    command(lmp, "change_box all x delta 0 " * string(lxEps) * " xy delta " * string(xyEps) * " xz delta " * string(xzEps) * " remap units box")
    command(lmp, "run 0")

    for i in 1:6
        C[i, 1] = -(LAMMPS.API.lammps_get_thermo(lmp, voigt_str_p[i]) * 1e-4 - S[i]) / eps
    end

    command(lmp, "change_box all x delta 0 " * string(-lxEps) * " xy delta " * string(-xyEps) * " xz delta " * string(-xzEps) * " remap units box")
    command(lmp, "run 0")

    command(lmp, "change_box all y delta 0 " * string(lyEps) * " yz delta " * string(yzEps) * " remap units box")
    command(lmp, "run 0")

    for i in 2:6
        C[i, 2] = -(LAMMPS.API.lammps_get_thermo(lmp, voigt_str_p[i]) * 1e-4 - S[i]) / eps
    end

    command(lmp, "change_box all y delta 0 " * string(-eps * LAMMPS.API.lammps_get_thermo(lmp, "ly")) *
                " yz delta " * string(-eps * LAMMPS.API.lammps_get_thermo(lmp, "yz")) * " remap units box")
    command(lmp, "run 0")

    command(lmp, "change_box all z delta 0 " * string(eps * LAMMPS.API.lammps_get_thermo(lmp, "lz")) *
                " remap units box")
    command(lmp, "run 0")

    for i in 3:6
        C[i, 3] = -(LAMMPS.API.lammps_get_thermo(lmp, voigt_str_p[i]) * 1e-4 - S[i]) / eps
    end

    command(lmp, "change_box all z delta 0 " * string(-eps * LAMMPS.API.lammps_get_thermo(lmp, "lz")) *
                " remap units box")
    command(lmp, "run 0")

    command(lmp, "change_box all yz delta " * string(eps * LAMMPS.API.lammps_get_thermo(lmp, "lz")) *
                " remap units box")
    command(lmp, "run 0")

    for i in 4:6
        C[i, 4] = -(LAMMPS.API.lammps_get_thermo(lmp, voigt_str_p[i]) * 1e-4 - S[i]) / eps
    end

    command(lmp, "change_box all yz delta " * string(-eps * LAMMPS.API.lammps_get_thermo(lmp, "lz")) *
                " remap units box")
    command(lmp, "run 0")

    command(lmp, "change_box all xz delta " * string(eps * LAMMPS.API.lammps_get_thermo(lmp, "lz")) *
                " remap units box")
    command(lmp, "run 0")

    for i in 5:6
        C[i, 5] = -(LAMMPS.API.lammps_get_thermo(lmp, voigt_str_p[i]) * 1e-4 - S[i]) / eps
    end

    command(lmp, "change_box all xz delta " * string(-eps * LAMMPS.API.lammps_get_thermo(lmp, "lz")) *
                " remap units box")
    command(lmp, "run 0")

    command(lmp, "change_box all xy delta " * string(eps * LAMMPS.API.lammps_get_thermo(lmp, "ly")) *
                " remap units box")
    command(lmp, "run 0")

    
    C[6,6] = -(LAMMPS.API.lammps_get_thermo(lmp, voigt_str_p[6]) * 1e-4 - S[6]) / eps

    command(lmp, "change_box all xy delta " * string(-eps * LAMMPS.API.lammps_get_thermo(lmp, "ly")) *
                " remap units box")
    command(lmp, "run 0")

    for i in 1:6
        for j in i+1:6
            C[i, j] = C[j, i]
        end
    end

    return S, C
end

F = zeros(3,3)
F[1,1] = F[2,2] = F[3,3] = 1
F[1,1] = 1 + 1e-1

timestep = 5e-3
strain_rate = 1e-3

# arve_new = apply_C_ARVE!(lmp_ref, arve_old, F, timestep, strain_rate)

get_stress(lmp_ref)*1e-4

command(lmp_ref, "write_dump all atom dump_tmp.atom")


num_increment = 20
dF = F^(1/num_increment)

F = zeros(3,3)
F[1,1] = F[2,2] = F[3,3] = 1

strain = Float64[]
push!(strain, F[1,1])

stress = get_stress(lmp_ref)*1e-4

s11 = Float64[]
push!(s11, stress[1])

for inc in 1:num_increment
    F = dF * F
    push!(strain, F[1,1])
    arve_new = apply_C_ARVE!(lmp_ref, arve_old, dF, timestep, strain_rate)
    arve_old = deepcopy(arve_new)
    stress = get_stress(lmp_ref)
    push!(s11, stress[1])
end

using Plots

plot(strain, s11)

nCell = 1
nQuads = 4

arves = Array{ARVE, 2}(undef, nCell, nQuads)
fill!(arves, arve_old)

for cell in 1:nCell
    for q in 1:nQuads
        set_ARVE_to_lmp!(lmp_ref, arves[cell, q]);
        @show get_stress(lmp_ref)*1e-4
    end
end

# FEM implementation, using Ferrite.jl package
using Ferrite
using FerriteAssembly

# Generate a grid
N = 1
L = 1.0
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


# Base.@kwdef struct ARVEmat{T} 

# end

LAMMPS.API.lammps_commands_string(lmp_ref, test_str)
command(lmp_ref, "run 1000")

function get_stress_gpa(lmp::LMP)
    return get_stress(lmp)*1e-4
end

arve_ref = get_ARVE_from_lmp(lmp_ref)

ARVE_State = Union{Tensor{2,3}, ARVE}


function FerriteAssembly.create_cell_state(::ARVE, cv::CellValues, args...)
    cell_state_qp = Vector{ARVE_State}(undef, 2)
    cell_state_qp[1] = one(Tensor{2,3})
    cell_state_qp[2] = arve_ref
    return [cell_state_qp for _ in 1:getnquadpoints(cv)]
end


function FerriteAssembly.element_residual!(re, state, ue, m::ARVE, cv::CellValues, buffer)
    Δt = FerriteAssembly.get_time_increment(buffer)
    old_states = FerriteAssembly.get_old_state(buffer)
    ndofs = getnbasefunctions(cv)
    for qp in 1:getnquadpoints(cv)
        old_state = old_states[qp]
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u # F is a Tensor
        set_ARVE_to_lmp!(lmp_ref, old_state[2])
        state[q][2] = apply_C_ARVE!(lmp_ref, old_state[2], F, timestep, strain_rate)
        # σ_voigt, C_voigt = calc_S_C(lmp_ref)
        σ_voigt = get_stress_gpa(lmp_ref)
        σ = fromvoigt(SymmetricTensor{2,3}, σ_voigt)
        P = det(F) * σ ⋅ transpose(inv(F))
        # Calculation ∂P∂F
        # C = fromvoigt(SymmetricTensor{4,3}, C_voigt)
        # Loop over test function
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            re[i] += (∇δui ⊡ P) * dΩ
            # ∇δui∂P∂F = ∇δui ⊡ ∂P∂F
            # for j in 1:ndofs
            #     ∇δuj = shape_gradient(cv, qp, j)
            #     ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            # end
        end
    end
end

residual = zeros(ndofs(dh))
u = zeros(ndofs(dh))
reAssembler = ReAssembler(residual)

# threading=true causes an error! because of multi-threading in LAMMPS
buffer = setup_domainbuffer(DomainSpec(dh, arve_ref, cv); threading=false)

work!(reAssembler, buffer; a=u)
