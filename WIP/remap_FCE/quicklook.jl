function quicklook_JW06(t, model, diags, state)
    if model isa TwinModels
        quicklook_JW06(t, model.HPE, diags.HPE, state.HPE)
        quicklook_JW06(t, model.FCE, diags.FCE, state.FCE)
    else
        session = open(diags; model, state, to_lonlat=identity)
        plot_lev("$(short_name(model)) at $(timeinfo(t/3600))", "ulat", -session.uv.ucolat, 10)
        plot_lev("$(short_name(model)) at $(timeinfo(t/3600))", "T", session.temperature, 10)
    end
end

function quicklook_DCMIP21(t, model, diags, state)
    if model isa TwinModels
        quicklook_DCMIP21(t, model.HPE, diags.HPE, state.HPE)
        quicklook_DCMIP21(t, model.FCE, diags.FCE, state.FCE)
    else
        session = open(diags; model, state, to_lonlat=identity)
        plot_Teq("$(short_name(model)) at $(timeinfo(t/3600))", session)
    end
end

function plot_Teq(name, session)
    T = session.temperature
    T = T[div(size(T,1),2),1:2:div(size(T,2),2), 1:16]
    T = T .- mean(T)
    plotmap(collect(T'), "$name : T-<T> at the Equator")
end

function plot_lev(name, varname, T, level)
    plotmap(fliplat(T[:,:,level]), "$name : $varname at model level $level")
end

short_name(::HPE) = "HPE"
short_name(::FCE) = "FCE"

slice(x) = transpose(x[div(size(x,1), 2), :,:])
fliplat(x) = reverse(x; dims=1)
Linf(x) = maximum(abs,x)
sym(x, op) = Linf(op(x,fliplat(x)))/Linf(x)
#plotmap(x::Matrix, title="") = display(heatmap(fliplat(x); title))
plotmap(x::Matrix, title="") = display(heatmap(x; title))
plotslice(x) = display(heatmap(slice(x)))

reshp(x) = reshape(x, sph.nlon, sph.nlat, size(x,2))
slice_Eq(x) = collect((x[div(size(x,1),2),:,:])')
