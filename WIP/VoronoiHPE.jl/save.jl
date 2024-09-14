function save(outputs, tape, filename)
    nt = length(tape)
    for i in 1:nt
        @info "Writing state $i"
        outs = outputs(tape[i])
        if i==1
            for (name, data) in outs
                vname = string(name)
                if data isa Matrix
                    ny, nx = size(data)
                    nccreate(filename, vname, "lon", nx, "lat", ny, "time", nt)
                else
                    nz, ny, nx = size(data)
                    nccreate(filename, vname, "lon", nx, "lat", ny, "lev", nz, "time", nt)
                end
            end
        end

        for (name, data) in outs
            vname = string(name)
            if data isa Matrix
                data = permutedims(data, (2,1))
                nx, ny = size(data)
                data = reshape(data, (nx,ny,1))
                ncwrite(data, filename, vname; start = [1, 1, i])
            else
                data = permutedims(data, (3,2,1))
                nx, ny, nz = size(data)
                data = reshape(data, (nx,ny,nz,1))
                ncwrite(data, filename, vname; start = [1, 1, 1, i])
            end
        end
    end
    ncclose(filename)
end
