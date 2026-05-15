from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.config import config
from nomad.datamodel.data import ArchiveSection, EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.basesections import Measurement, MeasurementResult
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection

# Import the reader package
from readers_ientrance.thermal_reader import read_thermal_dat

configuration = config.get_plugin_entry_point(
    'nomad_measurements_thermal.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()

# Constant for Unit Conversion (CGS to SI)
OE_TO_AM = 1000.0 / (4.0 * np.pi)  # Oersted to A/m


# ==========================================
# 1. THERMAL SAMPLE SECTION
# ==========================================
class ThermalSample(ArchiveSection):
    """Section for storing sample metadata extracted from the BEGIN:PARAMS block."""

    sample_id = Quantity(
        type=str, description='Unique identifier or name for the sample.'
    )
    sample_length = Quantity(type=np.float64, description='Length of the sample.')
    cell_constant = Quantity(
        type=np.float64, description='Cell constant used during measurement.'
    )
    offset_mode = Quantity(type=str, description='Offset mode setting.')
    dilation_offset = Quantity(type=np.float64, description='Dilation offset applied.')
    rotator_angle = Quantity(type=np.float64, description='Rotator angle setting.')
    sample_slot = Quantity(type=str, description='Slot where the sample is placed.')


# ==========================================
# 2. THERMAL RESULTS SECTION
# ==========================================
class ThermalResult(MeasurementResult):
    """Section for storing all the extracted arrays."""

    time_stamp = Quantity(
        type=np.float64, shape=['*'], unit='s',
        description='Time stamp array for each measurement point.'
    )
    comment = Quantity(
        type=str, shape=['*'],
        description='Inline comments or parameters recorded for each data point.'
    )
    system_temperature = Quantity(
        type=np.float64, shape=['*'], unit='K',
        description='Temperature of the measurement system over time.'
    )
    sample_temperature = Quantity(
        type=np.float64, shape=['*'], unit='K',
        description='Measured temperature of the sample over time.'
    )
    sample_temperature_rate = Quantity(
        type=np.float64, shape=['*'], unit='K/s',
        description='Rate of change of the sample temperature.'
    )
    sample_temperature_range = Quantity(
        type=np.float64, shape=['*'], unit='K',
        description='Temperature range setting for the sample.'
    )
    field = Quantity(
        type=np.float64, shape=['*'], unit='A/m',
        description='Applied magnetic field (converted from Oe to A/m).'
    )
    field_rate = Quantity(
        type=np.float64, shape=['*'], unit='A/m/s',
        description='Sweep rate of the applied magnetic field (converted from Oe/sec to A/m/s).'
    )
    chamber_pres = Quantity(
        type=np.float64, shape=['*'], unit='torr',
        description='Pressure inside the sample chamber.'
    )
    temperature_status = Quantity(
        type=np.float64, shape=['*'],
        description='Status code for the temperature controller.'
    )
    field_status = Quantity(
        type=np.float64, shape=['*'],
        description='Status code for the magnetic field controller.'
    )
    chamber_status = Quantity(
        type=np.float64, shape=['*'],
        description='Status code for the sample chamber environment.'
    )
    bridge_cycle = Quantity(
        type=np.float64, shape=['*'],
        description='Measurement cycle count for the readout bridge.'
    )
    rotator_angle = Quantity(
        type=np.float64, shape=['*'], unit='deg',
        description='Angle of the sample rotator during the measurement.'
    )
    therm_resistance = Quantity(
        type=np.float64, shape=['*'], unit='ohm',
        description='Measured thermal resistance.'
    )
    therm_resistance_rate = Quantity(
        type=np.float64, shape=['*'], unit='ohm/s',
        description='Rate of change of the thermal resistance.'
    )
    cell_imbalance = Quantity(
        type=np.float64, shape=['*'], unit='ppm',
        description='Imbalance measured across the measurement cell.'
    )
    cell_imbalance_rate = Quantity(
        type=np.float64, shape=['*'], unit='ppm/s',
        description='Rate of change of the cell imbalance.'
    )
    tap_imbalance = Quantity(
        type=np.float64, shape=['*'], unit='ppm',
        description='Tap imbalance measurement.'
    )
    coarse_dac_imbalance = Quantity(
        type=np.float64, shape=['*'], unit='ppm',
        description='Coarse Digital-to-Analog Converter (DAC) imbalance.'
    )
    fine_dac_imbalance = Quantity(
        type=np.float64, shape=['*'], unit='ppm',
        description='Fine Digital-to-Analog Converter (DAC) imbalance.'
    )
    loop_imbalance = Quantity(
        type=np.float64, shape=['*'], unit='ppm',
        description='Imbalance measured within the feedback loop.'
    )
    dilation = Quantity(
        type=np.float64, shape=['*'], unit='ppm',
        description='Measured dilation (thermal expansion) of the sample.'
    )
    dilation_rate = Quantity(
        type=np.float64, shape=['*'], unit='ppm/s',
        description='Rate of change of the sample dilation.'
    )
    therm_exp_coeff_raw = Quantity(
        type=np.float64, shape=['*'], unit='ppm/K',
        description='Raw, uncorrected thermal expansion coefficient.'
    )
    therm_exp_coeff = Quantity(
        type=np.float64, shape=['*'], unit='ppm/K',
        description='Final processed thermal expansion coefficient.'
    )
    therm_exp_coeff_compare = Quantity(
        type=np.float64, shape=['*'], unit='ppm/K',
        description='Thermal expansion coefficient used for comparison.'
    )
    therm_exp_coeff_diff_percentage = Quantity(
        type=np.float64, shape=['*'], unit='%',
        description='Percentage difference between sample and reference thermal expansion.'
    )
    therm_exp_coeff_diff_absolute = Quantity(
        type=np.float64, shape=['*'], unit='ppm/K',
        description='Absolute difference between sample and reference thermal expansion.'
    )
    therm_exp_coeff_baseline = Quantity(
        type=np.float64, shape=['*'], unit='ppm/K',
        description='Baseline thermal expansion coefficient of the system.'
    )
    therm_exp_coeff_reference = Quantity(
        type=np.float64, shape=['*'], unit='ppm/K',
        description='Reference thermal expansion coefficient for calibration.'
    )


# ==========================================
# 3. MAIN THERMAL ENTRY DATA
# ==========================================
class ThermalMeasurement(Measurement, EntryData):
    """Main EntryData schema triggered by the matching parser."""

    m_def = Section(
        a_eln=dict(lane_width='600px'),
    )

    data_file = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
        a_browser=dict(adaptor='RawFileAdaptor'),
        description='The uploaded raw data file (.dat/.txt) for this measurement.'
    )

    title = Quantity(
        type=str, description='Title extracted from the header of the raw file.'
    )
    datatype_time = Quantity(
        type=str, description='DATATYPE identifier for TIME extracted from the header.'
    )
    datatype_comment = Quantity(
        type=str, description='DATATYPE identifier for COMMENT extracted from the header.'
    )

    sample = SubSection(section_def=ThermalSample, repeats=True)
    results = SubSection(section_def=ThermalResult, repeats=True)

    def _map_sample(self, metadata: dict) -> None:
        """Helper method to map sample metadata."""
        if not self.sample:
            self.sample = [ThermalSample()]

        smp = self.sample[0]
        smp.sample_length = (
            float(metadata.get('sample_length'))
            if 'sample_length' in metadata
            else None
        )
        smp.cell_constant = (
            float(metadata.get('cell_constant'))
            if 'cell_constant' in metadata
            else None
        )
        smp.offset_mode = metadata.get('offset_mode')
        smp.dilation_offset = (
            float(metadata.get('dilation_offset'))
            if 'dilation_offset' in metadata
            else None
        )
        smp.rotator_angle = (
            float(metadata.get('rotator_angle'))
            if 'rotator_angle' in metadata
            else None
        )
        smp.sample_slot = metadata.get('sample_slot')

    def _map_results(self, thermal_data) -> None:
        """Helper method to map data arrays to the results section."""
        if not self.results:
            self.results = [ThermalResult()]

        res = self.results[0]
        res.time_stamp = (
            thermal_data.time_stamp if thermal_data.time_stamp is not None else None
        )
        res.system_temperature = (
            thermal_data.system_temperature
            if thermal_data.system_temperature is not None
            else None
        )
        res.sample_temperature = (
            thermal_data.sample_temperature
            if thermal_data.sample_temperature is not None
            else None
        )
        res.sample_temperature_rate = (
            thermal_data.sample_temperature_rate
            if thermal_data.sample_temperature_rate is not None
            else None
        )
        res.sample_temperature_range = (
            thermal_data.sample_temperature_range
            if thermal_data.sample_temperature_range is not None
            else None
        )

        # Apply SI Conversion to magnetic field fields
        res.field = (
            thermal_data.field * OE_TO_AM if thermal_data.field is not None else None
        )
        res.field_rate = (
            thermal_data.field_rate * OE_TO_AM
            if thermal_data.field_rate is not None
            else None
        )

        res.chamber_pres = (
            thermal_data.chamber_pres if thermal_data.chamber_pres is not None else None
        )
        res.temperature_status = (
            thermal_data.temperature_status
            if thermal_data.temperature_status is not None
            else None
        )
        res.field_status = (
            thermal_data.field_status if thermal_data.field_status is not None else None
        )
        res.chamber_status = (
            thermal_data.chamber_status
            if thermal_data.chamber_status is not None
            else None
        )
        res.bridge_cycle = (
            thermal_data.bridge_cycle if thermal_data.bridge_cycle is not None else None
        )
        res.rotator_angle = (
            thermal_data.rotator_angle
            if thermal_data.rotator_angle is not None
            else None
        )
        res.therm_resistance = (
            thermal_data.therm_resistance
            if thermal_data.therm_resistance is not None
            else None
        )
        res.therm_resistance_rate = (
            thermal_data.therm_resistance_rate
            if thermal_data.therm_resistance_rate is not None
            else None
        )
        res.cell_imbalance = (
            thermal_data.cell_imbalance
            if thermal_data.cell_imbalance is not None
            else None
        )
        res.cell_imbalance_rate = (
            thermal_data.cell_imbalance_rate
            if thermal_data.cell_imbalance_rate is not None
            else None
        )
        res.tap_imbalance = (
            thermal_data.tap_imbalance
            if thermal_data.tap_imbalance is not None
            else None
        )
        res.coarse_dac_imbalance = (
            thermal_data.coarse_dac_imbalance
            if thermal_data.coarse_dac_imbalance is not None
            else None
        )
        res.fine_dac_imbalance = (
            thermal_data.fine_dac_imbalance
            if thermal_data.fine_dac_imbalance is not None
            else None
        )
        res.loop_imbalance = (
            thermal_data.loop_imbalance
            if thermal_data.loop_imbalance is not None
            else None
        )
        res.dilation = (
            thermal_data.dilation if thermal_data.dilation is not None else None
        )
        res.dilation_rate = (
            thermal_data.dilation_rate
            if thermal_data.dilation_rate is not None
            else None
        )
        res.therm_exp_coeff_raw = (
            thermal_data.therm_exp_coeff_raw
            if thermal_data.therm_exp_coeff_raw is not None
            else None
        )
        res.therm_exp_coeff = (
            thermal_data.therm_exp_coeff
            if thermal_data.therm_exp_coeff is not None
            else None
        )
        res.therm_exp_coeff_compare = (
            thermal_data.therm_exp_coeff_compare
            if thermal_data.therm_exp_coeff_compare is not None
            else None
        )
        res.therm_exp_coeff_diff_percentage = (
            thermal_data.therm_exp_coeff_diff_percentage
            if thermal_data.therm_exp_coeff_diff_percentage is not None
            else None
        )
        res.therm_exp_coeff_diff_absolute = (
            thermal_data.therm_exp_coeff_diff_absolute
            if thermal_data.therm_exp_coeff_diff_absolute is not None
            else None
        )
        res.therm_exp_coeff_baseline = (
            thermal_data.therm_exp_coeff_baseline
            if thermal_data.therm_exp_coeff_baseline is not None
            else None
        )
        res.therm_exp_coeff_reference = (
            thermal_data.therm_exp_coeff_reference
            if thermal_data.therm_exp_coeff_reference is not None
            else None
        )
        res.comment = thermal_data.comment

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        """
        Triggered by the Parser. Handles reading the file and mapping output directly to the schema.
        """
        super().normalize(archive, logger)

        if not self.data_file:
            return

        logger.info('Parsing Thermal Measurement file', data_file=self.data_file)

        try:
            # Safely fetch the absolute path to the uploaded file
            with archive.m_context.raw_file(self.data_file, 'r') as f:
                file_path = f.name

            # Execute your standalone reader
            thermal_data = read_thermal_dat(file_path)

        except Exception as e:
            logger.error('Failed to parse thermal data file.', exc_info=e)
            return

        # 1. Map Top-Level Metadata
        self.title = thermal_data.metadata.get('TITLE')

        # 2. Map Sample Parameters
        self._map_sample(thermal_data.metadata)

        # 3. Map Results Arrays
        self._map_results(thermal_data)


m_package.__init_metainfo__()