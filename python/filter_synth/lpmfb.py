#!/usr/bin/python3
import sys
import numpy as np
import matplotlib.pyplot as plt

from PySpice.Unit import u_Ohm, u_kOhm, u_MOhm, u_F, u_V, u_us, u_ms
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Spice.Library import SpiceLibrary


xyce_install = '/home/ndx/XyceInstall/Parallel/bin/Xyce'
local_lib = SpiceLibrary('/home/ndx/dev/spin/src/spin/resources/models/')


class IdealOpAmp(SubCircuitFactory):
    NAME = 'IdealOpAmp'
    NODES = ['in+', 'in-', 'v+', 'v-', 'out']

    def __init__(self):
        super().__init__()

        # Input impedance
        self.R('input', 'in+', 'in-', 10@u_MOhm)

        # dc gain=100k and pole1=100hz
        # unity gain = dcgain x pole1 = 10MHZ
        self.VCVS('gain', 1, self.gnd, 'in+',
                  'in-', voltage_gain=kilo(100))
        self.R('P1', 1, 2, 1@u_kOhm)
        self.C('P1', 2, self.gnd, 1.5915@u_uF)

        # Output buffer and resistance
        self.VCVS('buffer', 3, self.gnd, 2, self.gnd, 1)
        self.R('out', 3, 'out', 10@u_Ohm)


class FilterTestRig(Circuit):
    def __init__(self, DUT, opamp, VCC, Rload=10@u_MOhm):
        self.DUT = DUT
        super().__init__(title=f'{self.DUT.NAME}_FTR')

        # Add op amp as a subcircuit
        if opamp == 'IdealOpAmp':
            self.subcircuit(IdealOpAmp())
        else:
            self.include(local_lib[opamp])

        # Add a positive rail, a mid-voltage point and a sine voltage source for AC analysis
        self.V(1, 'vcc', self.gnd, VCC)
        self.V(2, 'vmid', self.gnd, VCC/2)
        self.SinusoidalVoltageSource('input', 'vin', self.gnd, dc_offset=VCC/2)

        # Add the DUT as a subcircuit
        self.subcircuit(self.DUT)
        self.X(1, self.DUT.NAME, 'vin', 'out', 'vcc', self.gnd)

        # Load output
        self.R(f'Rl', 'out', self.gnd, Rload)

    def ac_analysis(self, start_frequency, stop_frequency, points_per_decade=300):
        simulator = self.simulator(
            'xyce-parallel', xyce_install, temperature=25, nominal_temperature=25)

        results = simulator.ac(start_frequency=start_frequency, stop_frequency=stop_frequency,
                               number_of_points=points_per_decade,  variation='dec')
        return results


def design_MFB_LP(f0, H, Q, C1):
    k = 2*np.pi*f0*C1
    C2 = 4*Q**2*(H+1)*C1
    R1 = 1 / (2*k*H*Q)
    R2 = 1 / (2*k*Q)
    R3 = 1 / (2*k*(H+1)*Q)

    return {"R1": R1, "R2": R2, "R3": R3, "C1": C1, "C2": C2}


def main(argv):
    values = design_MFB_LP(f0=12.34e3, H=1, Q=0.5, C1=1e-9)
    print(values)


if __name__ == '__main__':
    main(sys.argv[1:])
