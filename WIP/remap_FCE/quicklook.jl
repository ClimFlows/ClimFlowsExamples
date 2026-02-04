function quicklook(t, model, diags, state)
    if model isa TwinModels
        quicklook(t, model.HPE, diags.HPE, state.HPE)
        quicklook(t, model.FCE, diags.FCE, state.FCE)
    else
        lev = 20
        session = open(diags; model, state, to_lonlat=identity)
        plot_Teq("$(short_name(model)) at $(timeinfo(t/3600))", session)
    end
end

short_name(::HPE) = "HPE"
short_name(::FCE) = "FCE"

function plot_Teq(name, session)
    T = session.temperature
    T = T[div(size(T,1),2),1:2:div(size(T,2),2), 1:16]
    T = T .- mean(T)
    plotmap(collect(T'), "$name : T-<T> at the Equator")
end

# for quicklooks
slice(x) = transpose(x[div(size(x,1), 2), :,:])
fliplat(x) = reverse(x; dims=1)
Linf(x) = maximum(abs,x)
sym(x, op) = Linf(op(x,fliplat(x)))/Linf(x)
#plotmap(x::Matrix, title="") = display(heatmap(fliplat(x); title))
plotmap(x::Matrix, title="") = display(heatmap(x; title))
plotslice(x) = display(heatmap(slice(x)))

reshp(x) = reshape(x, sph.nlon, sph.nlat, size(x,2))
slice_Eq(x) = collect((x[div(size(x,1),2),:,:])')
