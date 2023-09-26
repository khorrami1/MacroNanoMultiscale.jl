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
command(lmp_ref, "write_dump all atom ARVE_10_relaxed.atom")
command(lmp_ref, "write_data ARVE_10_relaxed.data")

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

function apply_C_ARVE!(lmp::LMP, arve::ARVE, F, timestep, strain_rate)
    C = transpose(F)*F
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
    a = lx*[1.0,0.0,0.0]
    b = xy*[1.0,0.0,0.0] + ly*[0.0,1.0,0.0]
    c = xz*[1.0,0.0,0.0] + yz*[0.0,1.0,0.0] + lz*[0.0,0.0,1.0]
    corss_a_b = cross(a,b)
    dot_a_Ca = dot(a, C*a)
    temp_var = sqrt(dot(corss_a_b, invC*corss_a_b))
    lx_new = sqrt(dot_a_Ca)
    ly_new = detF * temp_var/sqrt(dot_a_Ca)
    lz_new = dot(corss_a_b, c)/temp_var
    xy_new =  dot(a, C*b)/sqrt(dot_a_Ca)
    xz_new =  dot(a, C*c)/sqrt(dot_a_Ca)
    yz_new =  dot(corss_a_b, invC*cross(a,c))/(temp_var*sqrt(dot_a_Ca))
    xhi_new = xlo + lx_new
    yhi_new = ylo + ly_new
    zhi_new = zlo + lz_new
    command(lmp, "fix fix_deform all deform 1 x final "*string(xlo)*" "*
        string(xhi_new)*" y final "*string(ylo)*" "*string(yhi_new)*
        " z final "*string(zlo)*" "*string(zhi_new)*" xy final "*string(xy_new)*
        " xz final "*string(xz_new)*" yz final "*string(yz_new)*" remap x units box")
    #delta = [lx_new - lx, ly_new - ly, lz_new - lz, xy_new - xy, xz_new - xz, yz_new - yz]
    # num_MD_steps = Int(ceil(maximum(delta) / (epsilon_rate * timestep)))
    num_MD_steps = Int(ceil(maximum(abs.(F - I)) / (strain_rate * timestep)))
    command(lmp, "run "*string(num_MD_steps))

    return deepcopy(get_ARVE_from_lmp(lmp))
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