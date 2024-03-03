using Ferrite

grid = generate_grid(Hexahedron, (10, 10, 10))
dh = DofHandler(grid)
add!(dh, :u, 3)
close!(dh)

u = zeros(ndofs(dh))

function update_grid!(dh::DofHandler, u::Vector)
    nNodes = getnnodes(dh.grid)
    for i in 1:nNodes
        dof_node = [3*i-2, 3*i-1, 3*i]
        u_node = u[dof_node]
        dh.grid.nodes[i] = Node((dh.grid.nodes[i].x[1] + u_node[1],
         dh.grid.nodes[i].x[2] + u_node[2], dh.grid.nodes[i].x[3] + u_node[3])) 
    end
end

update_grid!(dh, ones(ndofs(dh)))

dh.grid.nodes