import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.config import config
from nomad.datamodel.data import ArchiveSection, EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.basesections import Measurement, MeasurementResult
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection

# Import the reader package for DSC
from readers_ientrance import read_perkinelmer_dsc

# Import the reader package for TA Instruments DSC
from readers_ientrance.ta_dsc_reader import read_ta_dsc

# Import the reader package for Dilatometry
from readers_ientrance.thermal_reader import read_thermal_dat

configuration = config.get_plugin_entry_point(
    'nomad_measurements_thermal.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()

# Constant for Unit Conversion (CGS to SI)
OE_TO_AM = 1000.0 / (4.0 * np.pi)  # Oersted to A/m


# ==========================================
# 1. THERMAL SAMPLE SECTION (Dilatometry)
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
# 2. THERMAL RESULTS SECTION (Dilatometry)
# ==========================================
class ThermalResult(MeasurementResult):
    """Section for storing all the extracted arrays."""

    time_stamp = Quantity(
        type=np.float64,
        shape=['*'],
        unit='s',
        description='Time stamp array for each measurement point.',
    )
    comment = Quantity(
        type=str,
        shape=['*'],
        description='Inline comments or parameters recorded for each data point.',
    )
    system_temperature = Quantity(
        type=np.float64,
        shape=['*'],
        unit='K',
        description='Temperature of the measurement system over time.',
    )
    sample_temperature = Quantity(
        type=np.float64,
        shape=['*'],
        unit='K',
        description='Measured temperature of the sample over time.',
    )
    sample_temperature_rate = Quantity(
        type=np.float64,
        shape=['*'],
        unit='K/s',
        description='Rate of change of the sample temperature.',
    )
    sample_temperature_range = Quantity(
        type=np.float64,
        shape=['*'],
        unit='K',
        description='Temperature range setting for the sample.',
    )
    field = Quantity(
        type=np.float64,
        shape=['*'],
        unit='A/m',
        description='Applied magnetic field (converted from Oe to A/m).',
    )
    field_rate = Quantity(
        type=np.float64,
        shape=['*'],
        unit='A/m/s',
        description='Sweep rate of the applied magnetic field (converted from Oe/sec to A/m/s).',
    )
    chamber_pres = Quantity(
        type=np.float64,
        shape=['*'],
        unit='torr',
        description='Pressure inside the sample chamber.',
    )
    temperature_status = Quantity(
        type=np.float64,
        shape=['*'],
        description='Status code for the temperature controller.',
    )
    field_status = Quantity(
        type=np.float64,
        shape=['*'],
        description='Status code for the magnetic field controller.',
    )
    chamber_status = Quantity(
        type=np.float64,
        shape=['*'],
        description='Status code for the sample chamber environment.',
    )
    bridge_cycle = Quantity(
        type=np.float64,
        shape=['*'],
        description='Measurement cycle count for the readout bridge.',
    )
    rotator_angle = Quantity(
        type=np.float64,
        shape=['*'],
        unit='deg',
        description='Angle of the sample rotator during the measurement.',
    )
    therm_resistance = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ohm',
        description='Measured thermal resistance.',
    )
    therm_resistance_rate = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ohm/s',
        description='Rate of change of the thermal resistance.',
    )
    cell_imbalance = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm',
        description='Imbalance measured across the measurement cell.',
    )
    cell_imbalance_rate = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm/s',
        description='Rate of change of the cell imbalance.',
    )
    tap_imbalance = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm',
        description='Tap imbalance measurement.',
    )
    coarse_dac_imbalance = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm',
        description='Coarse Digital-to-Analog Converter (DAC) imbalance.',
    )
    fine_dac_imbalance = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm',
        description='Fine Digital-to-Analog Converter (DAC) imbalance.',
    )
    loop_imbalance = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm',
        description='Imbalance measured within the feedback loop.',
    )
    dilation = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm',
        description='Measured dilation (thermal expansion) of the sample.',
    )
    dilation_rate = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm/s',
        description='Rate of change of the sample dilation.',
    )
    therm_exp_coeff_raw = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm/K',
        description='Raw, uncorrected thermal expansion coefficient.',
    )
    therm_exp_coeff = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm/K',
        description='Final processed thermal expansion coefficient.',
    )
    therm_exp_coeff_compare = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm/K',
        description='Thermal expansion coefficient used for comparison.',
    )
    therm_exp_coeff_diff_percentage = Quantity(
        type=np.float64,
        shape=['*'],
        unit='%',
        description='Percentage difference between sample and reference thermal expansion.',
    )
    therm_exp_coeff_diff_absolute = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm/K',
        description='Absolute difference between sample and reference thermal expansion.',
    )
    therm_exp_coeff_baseline = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm/K',
        description='Baseline thermal expansion coefficient of the system.',
    )
    therm_exp_coeff_reference = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ppm/K',
        description='Reference thermal expansion coefficient for calibration.',
    )


# ==========================================
# 3. MAIN THERMAL ENTRY DATA (Dilatometry)
# ==========================================
class ThermalMeasurement(Measurement, EntryData):
    """Main EntryData schema triggered by the Dilatometry parser."""

    m_def = Section(
        a_eln=dict(lane_width='600px'),
    )

    data_file = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
        a_browser=dict(adaptor='RawFileAdaptor'),
        description='The uploaded raw data file (.dat/.txt) for this measurement.',
    )

    title = Quantity(
        type=str, description='Title extracted from the header of the raw file.'
    )
    datatype_time = Quantity(
        type=str, description='DATATYPE identifier for TIME extracted from the header.'
    )
    datatype_comment = Quantity(
        type=str,
        description='DATATYPE identifier for COMMENT extracted from the header.',
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
        super().normalize(archive, logger)

        if not self.data_file:
            return

        logger.info('Parsing Thermal Measurement file', data_file=self.data_file)

        try:
            with archive.m_context.raw_file(self.data_file, 'r') as f:
                file_path = f.name
            thermal_data = read_thermal_dat(file_path)
        except Exception as e:
            logger.error('Failed to parse thermal data file.', exc_info=e)
            return

        self.title = thermal_data.metadata.get('TITLE')
        self._map_sample(thermal_data.metadata)
        self._map_results(thermal_data)


# ==========================================
# 4. DSC DATA SUBSECTIONS
# ==========================================
class DSCCalibrationInformation(ArchiveSection):
    filename = Quantity(type=str, description='Calibration configuration filename.')
    date_time = Quantity(type=str, description='Calibration date and time.')


class DSCInitialConditions(ArchiveSection):
    temperature = Quantity(
        type=np.float64, unit='°C', description='Initial start temperature.'
    )
    purge_gas = Quantity(type=str, description='Purge gas used in the chamber.')
    purge_gas_rate = Quantity(type=str, description='Flow rate of the purge gas.')
    baseline_filename = Quantity(
        type=str, description='Filename for the applied baseline.'
    )
    end_condition = Quantity(
        type=str, description='Instrument condition after the run.'
    )
    total_points_in_run = Quantity(
        type=np.float64, description='Total number of data points collected.'
    )


class DSCManualTuneCalibration(ArchiveSection):
    date = Quantity(type=str, description='Date of manual tune calibration.')
    slope = Quantity(type=np.float64, description='Slope value from manual tuning.')
    coarse_balance = Quantity(type=np.float64, description='Coarse balance setting.')
    fine_balance = Quantity(type=np.float64, description='Fine balance setting.')


class DSCSmartScanCalibration(ArchiveSection):
    date = Quantity(type=str, description='Date of SmartScan calibration.')
    smartscan_enabled = Quantity(
        type=str, description='Indicates whether SmartScan was enabled.'
    )
    calibration_file = Quantity(
        type=str, description='Calibration file used for SmartScan.'
    )
    starting_temperature = Quantity(
        type=np.float64, unit='°C', description='Start temperature of SmartScan.'
    )
    ending_temperature = Quantity(
        type=np.float64, unit='°C', description='End temperature of SmartScan.'
    )
    number_of_steps = Quantity(
        type=np.float64, description='Number of steps in the SmartScan profile.'
    )


class DSCSampleTemperatureCalibration(ArchiveSection):
    date = Quantity(type=str, description='Date of sample temperature calibration.')
    reference = Quantity(
        type=str, shape=['*'], description='Reference materials (e.g., Indium).'
    )
    expected_temperature = Quantity(
        type=np.float64,
        shape=['*'],
        unit='°C',
        description='Expected transition temperature.',
    )
    measured_temperature = Quantity(
        type=np.float64,
        shape=['*'],
        unit='°C',
        description='Measured transition temperature.',
    )
    weight = Quantity(
        type=np.float64,
        shape=['*'],
        unit='mg',
        description='Weight of reference material.',
    )
    scan_rate = Quantity(
        type=np.float64,
        shape=['*'],
        unit='°C/min',
        description='Scan rate used during calibration.',
    )


class DSCFurnaceTemperatureCalibration(ArchiveSection):
    minimum = Quantity(
        type=np.float64,
        unit='°C',
        description='Minimum furnace calibration temperature.',
    )
    maximum = Quantity(
        type=np.float64,
        unit='°C',
        description='Maximum furnace calibration temperature.',
    )


class DSCFurnaceCalibrationComputed(ArchiveSection):
    date = Quantity(type=str, description='Date furnace calibration was computed.')
    setpoints = Quantity(
        type=np.float64,
        shape=['*'],
        unit='°C',
        description='Furnace calibration temperature setpoints.',
    )
    boundaries = Quantity(
        type=np.float64,
        shape=['*'],
        unit='°C',
        description='Computed furnace boundaries.',
    )
    y_double_prime = Quantity(
        type=np.float64,
        shape=['*'],
        description='Second derivative values for control loop.',
    )


class DSCHeatFlowCalibrationValues(ArchiveSection):
    reference = Quantity(
        type=str, shape=['*'], description='Reference materials used for heat flow.'
    )
    temperature = Quantity(
        type=np.float64, shape=['*'], unit='°C', description='Calibration temperatures.'
    )
    expected = Quantity(
        type=np.float64,
        shape=['*'],
        unit='J/g',
        description='Expected heat capacity/enthalpy.',
    )
    measured = Quantity(
        type=np.float64,
        shape=['*'],
        unit='J/g',
        description='Measured heat capacity/enthalpy.',
    )
    weight = Quantity(
        type=np.float64,
        shape=['*'],
        unit='mg',
        description='Weight of reference material.',
    )
    scan_rate = Quantity(
        type=np.float64,
        shape=['*'],
        unit='°C/min',
        description='Scan rate used for heat flow calibration.',
    )


class DSCHeatFlowCalibrationComputed(ArchiveSection):
    date = Quantity(type=str, description='Date heat flow calibration was computed.')
    k_ts = Quantity(type=str, description='Polynomial K(Ts) used to correct heat flow.')


class DSCProfileValues(ArchiveSection):
    software_version = Quantity(type=str, description='Software version used.')
    firmware_version = Quantity(type=str, description='Instrument firmware version.')
    instrument_serial_number = Quantity(type=str, description='Hardware serial number.')
    load_temperature = Quantity(
        type=np.float64, unit='°C', description='Sample load temperature.'
    )
    go_to_temp_rate = Quantity(
        type=np.float64, unit='°C/min', description='Rate used for Go To Temp commands.'
    )
    maximum_allowed_temperature = Quantity(
        type=np.float64,
        unit='°C',
        description='Maximum allowed instrument temperature.',
    )
    helium_purge = Quantity(
        type=str, description='Indicates if a Helium purge was used.'
    )
    liquid_nitrogen = Quantity(
        type=str, description='Indicates if Liquid Nitrogen cooling was used.'
    )
    data_taken_using_the = Quantity(type=str, description='Operational range applied.')
    filter_factor = Quantity(
        type=np.float64, description='Applied signal filtering factor.'
    )
    cooling_device = Quantity(
        type=str, description='Type of cooling accessory installed.'
    )
    wavelet_denoising_used = Quantity(
        type=str, description='Indicates if wavelet denoising was used.'
    )
    autoslope_used = Quantity(
        type=str, description='Indicates if autoslope correction was active.'
    )


class DSCResult(MeasurementResult):
    time = Quantity(
        type=np.float64, shape=['*'], description='Elapsed measurement time.'
    )
    unsubtracted_heat_flow = Quantity(
        type=np.float64,
        shape=['*'],
        description='Raw heat flow before baseline subtraction.',
    )
    baseline_heat_flow = Quantity(
        type=np.float64,
        shape=['*'],
        description='Measured or interpolated baseline heat flow.',
    )
    program_temperature = Quantity(
        type=np.float64, shape=['*'], description='Target program temperature.'
    )
    sample_temperature = Quantity(
        type=np.float64, shape=['*'], description='Actual measured sample temperature.'
    )
    approx_gas_flow = Quantity(
        type=np.float64,
        shape=['*'],
        description='Approximate flow rate of the purge gas.',
    )
    heat_flow_calibration = Quantity(
        type=np.float64,
        shape=['*'],
        description='Dynamic heat flow calibration multiplier.',
    )
    uncorrected_heat_flow = Quantity(
        type=np.float64,
        shape=['*'],
        description='Heat flow value prior to final correction.',
    )


class TADSCResult(MeasurementResult):
    """Dedicated DSC Results section for TA Instruments with explicit units."""

    time = Quantity(
        type=np.float64,
        shape=['*'],
        unit='min',
        description='Elapsed measurement time.',
    )
    sample_temperature = Quantity(
        type=np.float64,
        shape=['*'],
        unit='°C',
        description='Actual measured sample temperature.',
    )
    heat_flow = Quantity(
        type=np.float64, shape=['*'], unit='mW', description='Measured heat flow.'
    )
    heat_capacity = Quantity(
        type=np.float64,
        shape=['*'],
        unit='mJ/°C',
        description='Measured heat capacity.',
    )
    approx_gas_flow = Quantity(
        type=np.float64,
        shape=['*'],
        unit='ml/min',
        description='Flow rate of the sample purge gas.',
    )


# ==========================================
# 5. MAIN DSC ENTRY DATA
# ==========================================
class DSCMeasurement(Measurement, EntryData):
    m_def = Section(
        a_eln=dict(lane_width='600px'),
    )

    data_file = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
        a_browser=dict(adaptor='RawFileAdaptor'),
        description='The uploaded raw data file (.txt) for the DSC measurement.',
    )

    sample_id = Quantity(type=str, description='Name or identifier of the sample.')
    serial_number = Quantity(
        type=str, description='Serial number of the specific sample or batch.'
    )
    operator_id = Quantity(
        type=str, description='Name or identifier of the user operating the instrument.'
    )
    sample_weight = Quantity(
        type=np.float64, unit='mg', description='Mass of the sample.'
    )
    display_weight = Quantity(
        type=np.float64, unit='mg', description='Mass displayed/registered on the UI.'
    )
    data_collected = Quantity(
        type=str, description='Date and time the actual measurement began.'
    )
    comments = Quantity(
        type=str, description='Free text comments, often containing mass equations.'
    )

    validation_status = Quantity(
        type=str, description='Whether the data was validated.'
    )
    validation_by = Quantity(type=str, description='User who validated the data.')
    validation_date = Quantity(type=str, description='Date the data was validated.')

    calibration_information = SubSection(section_def=DSCCalibrationInformation)
    initial_conditions = SubSection(section_def=DSCInitialConditions)
    manual_tune_calibration = SubSection(section_def=DSCManualTuneCalibration)
    smartscan_calibration = SubSection(section_def=DSCSmartScanCalibration)
    sample_temperature_calibration = SubSection(
        section_def=DSCSampleTemperatureCalibration
    )
    furnace_temperature_calibration = SubSection(
        section_def=DSCFurnaceTemperatureCalibration
    )
    furnace_calibration_computed = SubSection(section_def=DSCFurnaceCalibrationComputed)
    heat_flow_calibration_values = SubSection(section_def=DSCHeatFlowCalibrationValues)
    heat_flow_calibration_computed = SubSection(
        section_def=DSCHeatFlowCalibrationComputed
    )
    profile_values = SubSection(section_def=DSCProfileValues)

    results = SubSection(section_def=DSCResult, repeats=True)

    def _extract_float(self, val: Any) -> float | None:
        """Helper to safely extract a float from a string containing text/units."""
        if not isinstance(val, str):
            return float(val) if val is not None else None
        match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val.replace(',', '.'))
        if match:
            return float(match.group())
        return None

    def _map_metadata_and_profile(self, dsc_data) -> None:
        """Extracts standard metadata and instrument profile values."""
        self.sample_id = dsc_data.metadata.get('Sample ID')
        self.serial_number = dsc_data.metadata.get('Serial Number')
        self.operator_id = dsc_data.metadata.get('Operator ID')
        self.sample_weight = self._extract_float(dsc_data.metadata.get('Sample Weight'))
        self.display_weight = self._extract_float(
            dsc_data.metadata.get('Display Weight')
        )
        self.data_collected = dsc_data.metadata.get('Data Collected')
        self.comments = dsc_data.metadata.get('Comment')

        if dsc_data.method_steps:
            self.description = 'Method Steps:\n' + '\n'.join(dsc_data.method_steps)

        self.validation_status = dsc_data.metadata.get('Validation_Validated')
        self.validation_by = dsc_data.metadata.get('Validation_By')
        self.validation_date = dsc_data.metadata.get('Validation_Date')

        pv = DSCProfileValues()
        pv.software_version = dsc_data.metadata.get(
            'PROFILE VALUES FOR THIS DATA_Software Version'
        )
        pv.firmware_version = dsc_data.metadata.get(
            'PROFILE VALUES FOR THIS DATA_Firmware Version'
        )
        pv.instrument_serial_number = dsc_data.metadata.get(
            'PROFILE VALUES FOR THIS DATA_Instrument Serial Number'
        )
        pv.load_temperature = self._extract_float(
            dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Load Temperature')
        )
        pv.maximum_allowed_temperature = self._extract_float(
            dsc_data.metadata.get(
                'PROFILE VALUES FOR THIS DATA_Maximum Allowed Temperature'
            )
        )
        self.profile_values = pv

    def _map_calibrations(self, dsc_data) -> None:
        """Extracts standard single-line calibration blocks."""
        cal_info = DSCCalibrationInformation()
        cal_info.filename = dsc_data.metadata.get('Calibration Information_Filename')
        cal_info.date_time = dsc_data.metadata.get('Calibration Information_Date/Time')
        self.calibration_information = cal_info

        init_cond = DSCInitialConditions()
        init_cond.temperature = self._extract_float(
            dsc_data.metadata.get('Initial Conditions_Temperature')
        )
        self.initial_conditions = init_cond

        mt_cal = DSCManualTuneCalibration()
        mt_cal.date = dsc_data.metadata.get('MANUAL TUNE CALIBRATION VALUES_Date')
        mt_cal.slope = self._extract_float(
            dsc_data.metadata.get('MANUAL TUNE CALIBRATION VALUES_Slope')
        )
        self.manual_tune_calibration = mt_cal

    def _map_tables(self, dsc_data) -> None:
        """Extracts table-based calibrations using declared index constants to avoid magic numbers."""
        # 1. Sample Temperature Calibration
        if 'SAMPLE TEMPERATURE CALIBRATION VALUES' in dsc_data.tables:
            st_cal = DSCSampleTemperatureCalibration()
            refs, exps, meas, wgts, scans = [], [], [], [], []

            # Using defined constants to satisfy Ruff PLR2004
            idx_ref, idx_exp, idx_meas, idx_wgt, idx_scan = 0, 1, 2, 3, 4

            for row in dsc_data.tables['SAMPLE TEMPERATURE CALIBRATION VALUES'][1:]:
                refs.append(row[idx_ref] if len(row) > idx_ref else '')
                exps.append(
                    self._extract_float(row[idx_exp]) if len(row) > idx_exp else None
                )
                meas.append(
                    self._extract_float(row[idx_meas]) if len(row) > idx_meas else None
                )
                wgts.append(
                    self._extract_float(row[idx_wgt]) if len(row) > idx_wgt else None
                )
                scans.append(
                    self._extract_float(row[idx_scan]) if len(row) > idx_scan else None
                )

            st_cal.reference = refs
            st_cal.expected_temperature = exps
            st_cal.measured_temperature = meas
            st_cal.weight = wgts
            st_cal.scan_rate = scans
            self.sample_temperature_calibration = st_cal

        # 2. Furnace Calibration Computed
        if 'FURNACE CALIBRATION COMPUTED' in dsc_data.tables:
            fc_comp = DSCFurnaceCalibrationComputed()
            sets, bounds, ys = [], [], []

            # Using defined constants to satisfy Ruff PLR2004
            idx_set, idx_bound, idx_y = 0, 1, 2

            for row in dsc_data.tables['FURNACE CALIBRATION COMPUTED'][1:]:
                sets.append(
                    self._extract_float(row[idx_set]) if len(row) > idx_set else None
                )
                bounds.append(
                    self._extract_float(row[idx_bound])
                    if len(row) > idx_bound
                    else None
                )
                ys.append(self._extract_float(row[idx_y]) if len(row) > idx_y else None)

            fc_comp.setpoints = sets
            fc_comp.boundaries = bounds
            fc_comp.y_double_prime = ys
            self.furnace_calibration_computed = fc_comp

        # 3. Heat Flow Calibration Values
        if 'HEAT FLOW CALIBRATION VALUES' in dsc_data.tables:
            hf_val = DSCHeatFlowCalibrationValues()
            refs, temps, exps, meas, wgts, scans = [], [], [], [], [], []

            # Using defined constants to satisfy Ruff PLR2004
            idx_ref, idx_temp, idx_exp, idx_meas, idx_wgt, idx_scan = 0, 1, 2, 3, 4, 5

            for row in dsc_data.tables['HEAT FLOW CALIBRATION VALUES'][1:]:
                refs.append(row[idx_ref] if len(row) > idx_ref else '')
                temps.append(
                    self._extract_float(row[idx_temp]) if len(row) > idx_temp else None
                )
                exps.append(
                    self._extract_float(row[idx_exp]) if len(row) > idx_exp else None
                )
                meas.append(
                    self._extract_float(row[idx_meas]) if len(row) > idx_meas else None
                )
                wgts.append(
                    self._extract_float(row[idx_wgt]) if len(row) > idx_wgt else None
                )
                scans.append(
                    self._extract_float(row[idx_scan]) if len(row) > idx_scan else None
                )

            hf_val.reference = refs
            hf_val.temperature = temps
            hf_val.expected = exps
            hf_val.measured = meas
            hf_val.weight = wgts
            hf_val.scan_rate = scans
            self.heat_flow_calibration_values = hf_val

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        if not self.data_file:
            return

        logger.info('Parsing DSC Measurement file', data_file=self.data_file)

        try:
            with archive.m_context.raw_file(self.data_file, 'r') as f:
                file_path = f.name
            dsc_data = read_perkinelmer_dsc(file_path)
        except Exception as e:
            logger.error('Failed to parse DSC data file.', exc_info=e)
            return

        # Use the newly split helper methods to keep this function clean and under the limit
        self._map_metadata_and_profile(dsc_data)
        self._map_calibrations(dsc_data)
        self._map_tables(dsc_data)

        if not self.results:
            self.results = [DSCResult()]

        res = self.results[0]
        res.time = dsc_data.time
        res.unsubtracted_heat_flow = dsc_data.unsubtracted_heat_flow
        res.baseline_heat_flow = dsc_data.baseline_heat_flow
        res.program_temperature = dsc_data.program_temperature
        res.sample_temperature = dsc_data.sample_temperature
        res.approx_gas_flow = dsc_data.approx_gas_flow
        res.heat_flow_calibration = dsc_data.heat_flow_calibration
        res.uncorrected_heat_flow = dsc_data.uncorrected_heat_flow


# ==========================================
# 6. TA INSTRUMENTS DSC MEASUREMENT
# ==========================================
class TADSCMeasurement(Measurement, EntryData):
    """Standalone entry schema for TA Instruments DSC text files."""

    m_def = Section(a_eln=dict(lane_width='600px'))

    data_file = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
        a_browser=dict(adaptor='RawFileAdaptor'),
        description='The uploaded raw data file (.txt) for the TA DSC measurement.',
    )

    # Core Metadata Fields
    sample_id = Quantity(type=str, description='Name or identifier of the sample.')
    operator_id = Quantity(
        type=str, description='Name or identifier of the user operating the instrument.'
    )
    sample_weight = Quantity(
        type=np.float64, unit='mg', description='Mass of the sample.'
    )
    data_collected = Quantity(
        type=str, description='Date and time the actual measurement began.'
    )
    comments = Quantity(type=str, description='Free text comments from the metadata.')

    # TA Specific Metadata Fields
    instrument = Quantity(type=str, description='Instrument model and version string.')
    module = Quantity(
        type=str, description='Instrument module compartment description.'
    )
    mode = Quantity(type=str, description='Measurement mode configuration.')
    method = Quantity(type=str, description='Configured run method type.')
    pan_mass = Quantity(
        type=str, description='Recorded mass values for the sample pan and lid.'
    )
    cell_constant = Quantity(
        type=np.float64, description='Calculated cell constant value (Kcell).'
    )
    cooling_unit = Quantity(
        type=str, description='Type of cooling accessory unit attached.'
    )

    # Calibration File Sub-fields
    calibration_file_tzero = Quantity(
        type=str, description='Path to the applied Tzero calibration file.'
    )
    calibration_file_baseline = Quantity(
        type=str, description='Path to the applied Baseline calibration file.'
    )
    calibration_file_sapphire = Quantity(
        type=str, description='Path to the applied Sapphire calibration file.'
    )

    results = SubSection(section_def=TADSCResult, repeats=True)

    def _extract_float(self, val: Any) -> float | None:
        """Local helper to safely extract a float from strings containing units."""
        if not isinstance(val, str):
            return float(val) if val is not None else None
        match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val.replace(',', '.'))
        if match:
            return float(match.group())
        return None

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        if not self.data_file:
            return

        logger.info(
            'Parsing TA Instruments DSC Measurement file', data_file=self.data_file
        )

        try:
            with archive.m_context.raw_file(self.data_file, 'r') as f:
                file_path = f.name
            dsc_data = read_ta_dsc(file_path)
        except Exception as e:
            logger.error('Failed to parse TA DSC data file.', exc_info=e)
            return

        # Map Metadata Fields
        self.sample_id = dsc_data.metadata.get('Sample')
        self.operator_id = dsc_data.metadata.get('Operator')
        self.sample_weight = self._extract_float(dsc_data.metadata.get('Size'))
        self.comments = dsc_data.metadata.get('Comment')

        date_str = dsc_data.metadata.get('Date', '')
        time_str = dsc_data.metadata.get('Time', '')
        self.data_collected = f'{date_str} {time_str}'.strip()

        # Combine OrgMethod recipe lines into the built-in description field
        if dsc_data.method_steps:
            self.description = 'Method Steps:\n' + '\n'.join(dsc_data.method_steps)

        # Map TA-Specific instrument settings
        self.instrument = dsc_data.metadata.get('Instrument')
        self.module = dsc_data.metadata.get('Module')
        self.mode = dsc_data.metadata.get('Mode')
        self.method = dsc_data.metadata.get('Method')
        self.pan_mass = dsc_data.metadata.get('PanMass')
        self.cell_constant = self._extract_float(dsc_data.metadata.get('Kcell'))
        self.cooling_unit = dsc_data.metadata.get('CoolingUnit')

        # Cleanly isolate the duplicate-keyed calibration paths
        cal_paths = dsc_data.metadata.get('InstCalFile', '')
        if 'Tzero:' in cal_paths:
            match = re.search(r'Tzero:\s*([^|]+)', cal_paths)
            if match:
                self.calibration_file_tzero = match.group(1).strip()
        if 'Baseline:' in cal_paths:
            match = re.search(r'Baseline:\s*([^|]+)', cal_paths)
            if match:
                self.calibration_file_baseline = match.group(1).strip()
        if 'Sapphire:' in cal_paths:
            match = re.search(r'Sapphire:\s*([^|]+)', cal_paths)
            if match:
                self.calibration_file_sapphire = match.group(1).strip()

        # Map unit-aware data arrays
        if not self.results:
            self.results = [TADSCResult()]

        res = self.results[0]
        res.time = dsc_data.time
        res.sample_temperature = dsc_data.sample_temperature
        res.heat_flow = dsc_data.heat_flow
        res.heat_capacity = dsc_data.heat_capacity
        res.approx_gas_flow = dsc_data.approx_gas_flow


m_package.__init_metainfo__()
