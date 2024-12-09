# Extracting Mobility from Transient Photoconductivity (TPC)

This is the code that was initially developed for: J. Lim & M. Kober-Czerny, Y.H. Lin, J.M. Ball, N. Sakai, E.A. Duijnstee, M.J. Hong, J.G. Labram, B. Wenger, H.J. Snaith , "Long-range charge carrier mobility in metal halide perovskite thin-films and single crystals via transient photo-conductivity". [doi:10.1038/s41467-022-31569-w](https://doi.org/10.1038/s41467-022-31569-w) <br>
The theory is described in the publication in more detail.


## Getting Started
### Installation and Usage
Install the [Gooey](https://pypi.org/project/Gooey/) package in any or a new Python (version 3.10) environment.
Open your python command line and run
```
python TPC_Fitting2-0.py
```
This opens a GUI to set up the analysis.

### Explanation of Settings

#### Main
<table>
  <tr>
    <th>Setting</th>
    <th>Explanation</th>
  </tr>
  <tr>
    <td>data_path</td>
    <td>Path to the datafile. Only one file can be selected at a time</td>
  </tr>
  <tr>
    <td>Filter_Wheel</td>
    <td>optical density of an additional filter that may have been used between the laser and the sample (except the automatic filter wheel in our setup)</td>
  </tr>
  <tr>
    <td>Voltage</td>
    <td>measured voltage of the battery used</td>
  </tr>
  <tr>
    <td>Thickness</td>
    <td>sample thickness in nm</td>
  </tr>
  <tr>
    <td>Sample_Name (optional)</td>
    <td>Do you want a sepcific sample name to be displayed on the figure?</td>
  </tr>
</table>

#### Electronic Parameters 
<table>
  <tr>
    <th>Setting</th>
    <th>Explanation</th>
  </tr>
  <tr>
    <td>Mask</td>
    <td>Which of the three pixels was used for the measurement? Select 'Single Crystal' for custom dimensions later</td>
  </tr>
  <tr>
    <td>Vertical</td>
    <td>Was the measurement done using the vertical mask? This has dimensions 1 mm x 1mm x thickness (nm)</td>
  </tr>
</table>

#### Optical Parameters
<table>
  <tr>
    <th>Setting</th>
    <th>Explanation</th>
  </tr>
  <tr>
    <td>laser_wavelength</td>
    <td>Peak wavelength of the used laser in nm</td>
  </tr>
  <tr>
    <td>laser_reference_file</td>
    <td>Select the latest laser calibration file. This file encodes the laser power for each wavelength</td>
  </tr>
  <tr>
    <td>Absorption_Coefficient</td>
    <td>absorption coefficient of the sample at the incident laser wavelength in cm^-1</td>
  </tr>
  <tr>
    <td>Reflectance</td>
    <td>reflectance of the sample at the incident laser wavelength in %</td>
  </tr>
</table>

#### Corrections
### Settings: Main
This is an explanation of all the settings needed in the 'Main' window

<table>
  <tr>
    <th>Setting</th>
    <th>Explanation</th>
  </tr>
  <tr>
    <td>Exciton_Binding_Energy</td>
    <td>exciton binding energy of the material in meV</td>
  </tr>
  <tr>
    <td>k1</td>
    <td>monomolecular recombination rate of sample in x 10^6 s^-1. If empty, this one will be estimated from the 'tail' of the TPC decay.</td>
  </tr>
  <tr>
    <td>k2</td>
    <td>bimolecular recombination rate of material in x 10^-10 cm^3 s^-1. If empty, this one will be estimated from intensity-dependent mobility (which we expect to be unchanged)</td>
  </tr>
  <tr>
    <td>k3</td>
    <td>Auger recombination rate of material in x 10^-28 cm^6 s^-1.</td>
  </tr>
</table>

#### Single Crystal Measures (only relevant if 'Single Crystal' set in Electronic Parameters)
### Settings: Main
This is an explanation of all the settings needed in the 'Main' window

<table>
  <tr>
    <th>Setting</th>
    <th>Explanation</th>
  </tr>
  <tr>
    <td>Single_Crystal_Witdh</td>
    <td>width of electrodes in mm</td>
  </tr>
  <tr>
    <td>Single_Crystal_Distance</td>
    <td>separation of electrodes in mm</td>
  </tr>
  <tr>
    <td>Single_Crystal_Thickness</td>
    <td>thickness of sample in mm</td>
  </tr>
</table>

### Analysis
Once all the settings are given, press 'Start'. The output will be a figure, which displays (a) Photoconductivity transients with the monoexponential fits, (b) the estimated lifetimes from fitting the decays and a reported, estimated k1, (c) the extracted peak photoconductivities, dark conductivities, and corrected conductivities, and (d) the extracted and corrected long-range sum-mobilities. <br>
In the GUI window and estimate for k2 is given, unless a k2 has been specified.