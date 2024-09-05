def _plot_energy_levels(array, cutoff=0, energy_unit='wavenumber', ):
    from matplotlib.pyplot import subplots
    from slothpy._general_utilities._ploting_utilities import energy_units

    unit_conversion, unit_label = energy_units(energy_unit)

    if cutoff == 0:
        pass
    else:
        array = array[0:cutoff]
    xdata = array * unit_conversion

    fig, ax = subplots()
    ax.scatter([0 for i in xdata], xdata, marker='_', linewidths=2, s=500, color='#000000')
    for energy in xdata:
        ax.text(0 + 0.004, energy, f'{energy:.2f} {unit_label}', va='center', ha='left')
    ax.set_xticks([])
    ax.set_ylabel(f'Energy / {unit_label}')
    fig.tight_layout()
    
    return fig, ax
