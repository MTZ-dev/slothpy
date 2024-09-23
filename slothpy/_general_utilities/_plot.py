def _plot_energy_levels(array, cutoff=0, energy_unit='wavenumber', marker_size=500, marker_width=2, marker_color='#000000', decimals=2, \
                        frame_width=1, frame_color='#000000', tick_length=6, tick_width=1, tick_direction='out', tick_colour='#000000', \
                        auto_ticks_locator=True, x_ticks=1, y_ticks=1, minor_ticks_frequency=2):
    from matplotlib.pyplot import subplots, setp
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    from slothpy._general_utilities._ploting_utilities import energy_units

    unit_conversion, unit_label = energy_units(energy_unit)

    if cutoff == 0:
        pass
    else:
        array = array[0:cutoff]
    xdata = array * unit_conversion

    fig, ax = subplots()
    ax.scatter([0 for i in xdata], xdata, marker='_', linewidths=marker_width, s=marker_size, color=marker_color)
    for energy in xdata:
        ax.text(0 + 0.004, energy, f'{energy:.{decimals}f} {unit_label}', va='center', ha='left')
    ax.set_xticks([])
    ax.set_ylabel(f'Energy / {unit_label}')
    setp(ax.spines.values(), linewidth=frame_width, color=frame_color)
    if not auto_ticks_locator:
        ax.yaxis.set_major_locator(MultipleLocator(y_ticks))
        ax.xaxis.set_major_locator(MultipleLocator(x_ticks))
    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks_frequency))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_ticks_frequency))
    ax.tick_params(axis='both', which='major', length=tick_length, width=tick_width, direction=tick_direction, color=tick_colour)
    ax.tick_params(axis='both', which='minor', length=tick_length*0.5, width=tick_width*0.5, direction=tick_direction, color=tick_colour)
    fig.tight_layout()
    
    return fig, ax
