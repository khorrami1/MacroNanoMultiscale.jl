using NearestNeighbors

function compute_nye_tensor(system1::Array{Float64,2}, system2::Array{Float64,2}, cutoff_radius::Float64)
    # Compute the Nye tensor for each atom in system2
    # system1: reference system (bulk with no defect)
    # system2: system to analyze
    # cutoff_radius: cutoff radius for neighbor search

    # Number of atoms in each system
    natoms1 = size(system1, 1)
    natoms2 = size(system2, 1)

    # Initialize Nye tensor array
    nye_tensor = Array{Float64}(undef, natoms2, 3, 3)

    # Create kdtree for neighbor search
    kdtree = KDTree(system1[:,1:3])

    for i in 1:natoms2
        # Find neighbors within cutoff radius
        idxs = inrange(kdtree, reshape(system2[i,1:3], (1,:)), cutoff_radius)

        # Compute lattice correspondence tensor G
        Q = system2[idxs,1:3]
        P = system1[idxs,:]
        G = inv(Q' * Q) * Q' * P

        # Compute Nye tensor
        nye_tensor[i,:,:] = 0.5 * (G' - inv(G))
    end

    return nye_tensor
end


using LinearAlgebra

# Define lattice constant and elastic constants for aluminum
alat = 4.05
C11 = 106.75
C12 = 60.41
C44 = 28.34

# Create perfect FCC lattice
perfect = [0.0 0.0 0.0; 0.5 0.5 0.0; 0.5 0.0 0.5; 0.0 0.5 0.5] * alat

# Create edge dislocation
nx = 10
ny = 10
nz = 10
disloc = zeros(nx*ny*nz*4,3)
for i in 1:nx
    for j in 1:ny
        for k in 1:nz
            idx = ((k-1)*ny + (j-1))*nx + i
            disloc[4*(idx-1)+1,:] = [i-1 j-1 k-1] * alat
            disloc[4*(idx-1)+2,:] = [i-1+0.5 j-1+0.5 k-1] * alat
            disloc[4*(idx-1)+3,:] = [i-1+0.5 j-1 k-1+0.5] * alat
            disloc[4*(idx-1)+4,:] = [i-1 j-1+0.5 k-1+0.5] * alat
        end
    end
end

# Compute Nye tensor for edge dislocation in FCC lattice
nye_tensor = compute_nye_tensor(perfect, disloc, alat)

# Plot Nye tensor component alpha_12 for edge dislocation in FCC lattice
using Plots
x = disloc[:,1]
y = disloc[:,2]
z = nye_tensor[:,1,2]
scatter(x,y,zcolor=z,title="Nye tensor component α_12 for edge dislocation in FCC lattice",xlabel="x",ylabel="y")
