
function default_plot_theme()
    thefont="Latin Modern Roman"
    majorfontsize = 24*2
    minorfontsize = 18*2
    tinyfontsize = 16*2
    T = Theme(
        Axis = (
            xlabelfont=thefont,
            ylabelfont=thefont,
            xticklabelfont=thefont,
            yticklabelfont=thefont,
            titlefont=thefont,
            xlabelsize=minorfontsize,
            ylabelsize=minorfontsize,
            xticklabelsize=tinyfontsize,
            yticklabelsize=tinyfontsize,
            titlesize=majorfontsize,
            topspinevisible = false,
            rightspinevisible = false,
            xgridvisible = false,
            ygridvisible = false,
        ),
        Scatter = (
            markersize=14,
        ),
        Lines = (
            linewidth = 3,
        ),
        Legend = (
            labelfont=thefont,
            labelsize=tinyfontsize
        ),
        Colorbar = (
            ticklabelfont=thefont, 
            ticklabelsize=tinyfontsize
        )
    )
    set_theme!(T)
end

function pitexlabel(n) 
    iszero(n) && return L"0"
    isone(n) && return L"\pi"
    isone(-n) && return L"-\pi"
    return L"%$n\pi"
end
function pilabel(n) 
    iszero(n) && return "0"
    isone(n) && return "π"
    isone(-n) && return "-π"
    return "$(n)π"
end
piticks(step, max=1) = (range(-max*pi, max*pi,step=step*pi), map(pitexlabel, -max:step:max)) 
