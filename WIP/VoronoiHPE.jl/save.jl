module Save

using NetCDF

function save(outputs, tape, filename)
    nt = length(tape)
    for i in 1:nt
        @info "Writing state $i to $filename"
        outs = outputs(tape[i])
        if i==1
            for (name, data) in outs
                vname = string(name)
                if data isa Matrix
                    ny, nx = size(data)
                    nccreate(filename, vname, "lon", nx, "lat", ny, "time", nt)
                else
                    ny, nx, nz = size(data)
                    nccreate(filename, vname, "lon", nx, "lat", ny, "lev_$vname", nz, "time", nt)
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
                data = permutedims(data, (2,1,3))
                nx, ny, nz = size(data)
                data = reshape(data, (nx,ny,nz,1))
                try
                    ncwrite(data, filename, vname; start = [1, 1, 1, i])
                catch
                    @error "Error while writing field $vname into $filename." size(data)
                    rethrow()
                end
            end
        end
    end
    ncclose(filename)
end

end

using .Save: save