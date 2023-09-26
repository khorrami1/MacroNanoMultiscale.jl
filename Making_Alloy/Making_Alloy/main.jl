
using LAMMPS

Pdamp = 100

commands = "
# Initialize
units metal
dimension 3
boundary p p p
atom_style atomic

# Read data file
read_data data.fcc
change_box all triclinic 

# Potential
pair_style eam/alloy
pair_coeff * * FeNiCrCoAl-heaweight.setfl Fe Ni Cr Co Al

# Settings
compute eng all pe/atom
compute eatoms all reduce sum c_eng

# Run
fix 1 all npt temp 1.0 1.0 1.0 x 0.0 0.0 $Pdamp y 0.0 0.0 $Pdamp z 0.0 0.0 $Pdamp xy 0.0 0.0 $Pdamp xz 0.0 0.0 $Pdamp yz 0.0 0.0 $Pdamp
thermo 1000
thermo_style custom step temp pxx pyy pzz pxy pxz pyz  
run 50000
"

lmp = LMP()
command(lmp, "package omp 4")
command(lmp, "suffix omp")

LAMMPS.API.lammps_commands_string(lmp, commands)


command(lmp, "write_restart alloy.equil")
command(lmp, "write_dump all atom dump.txt")


# If reastart file is available!

using LAMMPS
using LinearAlgebra

Pdamp = 100
timestep = 5e-3

commends_after_resstart = "
units metal
dimension 3
boundary p p p
atom_style atomic
read_restart alloy.equil
# Potential
pair_style eam/alloy
pair_coeff * * FeNiCrCoAl-heaweight.setfl Fe Ni Cr Co Al

# Settings
compute eng all pe/atom
compute eatoms all reduce sum c_eng

timestep $timestep

# Run
#fix 1 all npt temp 1.0 1.0 1.0 x 0.0 0.0 $Pdamp y 0.0 0.0 $Pdamp z 0.0 0.0 $Pdamp xy 0.0 0.0 $Pdamp xz 0.0 0.0 $Pdamp yz 0.0 0.0 $Pdamp
#thermo 1000
thermo_style custom step temp pxx pyy pzz pxy pxz pyz  
"



lmp = LMP()
# lmp = LMP(["-pk","omp", "12", "-sf", "omp"])
command(lmp, "package omp 4")
command(lmp, "suffix omp")

LAMMPS.API.lammps_commands_string(lmp, commends_after_resstart)

# command(lmp, "run 1")
# LAMMPS.API.lammps_close(lmp)


function apply_C_RVE!(lmp, F, timestep, epsilon_rate)
    C = transpose(F)*F
    detF = sqrt(det(C))
    invC = inv(C)
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
    delta_x = lx_new - lx
    delta_y = ly_new - ly
    delta_z = lz_new - lz
    command(lmp, "fix fix_deform all deform 1 x delta 0 $delta_x y delta 0 $delta_y z delta 0 $delta_z xy final $xy_new xz final $xz_new yz final $yz_new remap x units box")
    num_MD_steps = Int(ceil(maximum(abs.(F - I)) / (epsilon_rate * timestep)))
    println("num_MD_steps = $num_MD_steps")
    command(lmp, "run "*string(num_MD_steps))
end



epsilon_rate = 1e-3
command(lmp, "thermo 100")
command(lmp, "thermo_style custom step lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz ")
F = 1.0*Matrix(I, 3,3)
F[1,1] = 1.01
apply_C_RVE!(lmp, F, timestep, epsilon_rate)

command(lmp, "write_dump all atom dump_after_npt.atom")


function get_stress_lmp(lmp)
    Pxx = LAMMPS.API.lammps_get_thermo(lmp, "pxx")
    Pyy = LAMMPS.API.lammps_get_thermo(lmp, "pyy")
    Pzz = LAMMPS.API.lammps_get_thermo(lmp, "pzz")
    Pxy = LAMMPS.API.lammps_get_thermo(lmp, "pxy")
    Pxz = LAMMPS.API.lammps_get_thermo(lmp, "pxz")
    Pyz = LAMMPS.API.lammps_get_thermo(lmp, "pyz")
    return [-Pxx, -Pyy, -Pzz, -Pxy, -Pxz, -Pyz]*1e-4 # units: GPa
end

stress = get_stress_lmp(lmp)

function copy_lmp_2_lmp!(lmp_old, lmp_new)
    x_new = extract_atom(lmp_new, "x")
    v_new = extract_atom(lmp_new, "v")
    xlo = LAMMPS.API.lammps_get_thermo(lmp_new, "xlo")
    ylo = LAMMPS.API.lammps_get_thermo(lmp_new, "ylo")
    zlo = LAMMPS.API.lammps_get_thermo(lmp_new, "zlo")
    xhi = LAMMPS.API.lammps_get_thermo(lmp_new, "xhi")
    yhi = LAMMPS.API.lammps_get_thermo(lmp_new, "yhi")
    zhi = LAMMPS.API.lammps_get_thermo(lmp_new, "zhi")
    xy = LAMMPS.API.lammps_get_thermo(lmp_new, "xy")
    xz = LAMMPS.API.lammps_get_thermo(lmp_new, "xz")
    yz = LAMMPS.API.lammps_get_thermo(lmp_new, "yz")
    command(lmp_old, "change_box all x final "*string(xlo)*" "*
    string(xhi)*" y final "*string(ylo)*" "*string(yhi)*
    " z final "*string(zlo)*" "*string(zhi)*" xy final "*string(xy)*
    " xz final "*string(xz)*" yz final "*string(yz))
    LAMMPS.API.lammps_scatter_atoms(lmp_old, "x", 1, 3, x_new)
    LAMMPS.API.lammps_scatter_atoms(lmp_old, "v", 1, 3, v_new)
end

lmp2 = LMP()
command(lmp2, "package omp 4")
command(lmp2, "suffix omp")
LAMMPS.API.lammps_commands_string(lmp2, commends_after_resstart)

copy_lmp_2_lmp!



