using Random

# Define the lattice constant
a = 3.56

# Define the number of unit cells in each direction
nx = 4
ny = 4
nz = 4

# Define the concentration and mass of each element

rand_5 = rand(5)
conc_val = rand_5./sum(rand_5)

elements = [:Fe, :Ni, :Cr, :Co, :Al]
conc = Dict(:Fe => conc_val[1], :Ni => conc_val[2], :Cr => conc_val[3], :Co => conc_val[4], :Al => conc_val[5])
mass = Dict(:Fe => 55.845, :Ni => 58.6934, :Cr => 51.9961, :Co => 58.933195, :Al => 26.9815385)

# Create an array to store the atom types
atomtypes = Array{Int}(undef, 4*nx*ny*nz)

# Fill the array with atom types according to the concentration
cumulative_conc = cumsum(values(conc))
cumulative_conc ./= cumulative_conc[end]
for i in 1:length(atomtypes)
    r = rand()
    for (j, el) in enumerate(keys(conc))
        if r <= cumulative_conc[j]
            atomtypes[i] = j
            break
        end
    end
end

# Open a file for writing
open("data_small.fcc", "w") do f
    # Write the LAMMPS header
    println(f, "LAMMPS data file for FCC alloy")
    println(f)
    println(f, "$(length(atomtypes)) atoms")
    println(f)
    println(f, "$(length(elements)) atom types")
    println(f)
    println(f, "0.0 $(a*nx) xlo xhi")
    println(f, "0.0 $(a*ny) ylo yhi")
    println(f, "0.0 $(a*nz) zlo zhi")
    println(f)
    
    # Write the Masses section
    println(f, "Masses")
    println(f)
    
    for (i, el) in enumerate(elements)
        println(f, "$i $(mass[el])")
    end
    
    println(f)
    
    # Write the Atoms section
    println(f, "Atoms")
    println(f)
    
    i = 1
    for z in 0:nz-1
        for y in 0:ny-1
            for x in 0:nx-1
                # Write the corner atom
                println(f, "$i $(atomtypes[i]) $(x*a) $(y*a) $(z*a)")
                i += 1
                
                # Write the face-centered atoms
                println(f, "$i $(atomtypes[i]) $(x*a+a/2) $(y*a+a/2) $(z*a)")
                i += 1
                println(f, "$i $(atomtypes[i]) $(x*a) $(y*a+a/2) $(z*a+a/2)")
                i += 1
                println(f, "$i $(atomtypes[i]) $(x*a+a/2) $(y*a) $(z*a+a/2)")
                i += 1
            end
        end
    end
    
end

println("Done!")
