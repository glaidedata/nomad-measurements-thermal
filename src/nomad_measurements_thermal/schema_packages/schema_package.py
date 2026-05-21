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

# Import the reader package for Dilatometry
from readers_ientrance.thermal_reader import read_thermal_dat
# Import the reader package for DSC
from readers_ientrance import read_perkinelmer_dsc

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
            float(metadata.get('sample_length')) if 'sample_length' in metadata else None
        )
        smp.cell_constant = (
            float(metadata.get('cell_constant')) if 'cell_constant' in metadata else None
        )
        smp.offset_mode = metadata.get('offset_mode')
        smp.dilation_offset = (
            float(metadata.get('dilation_offset')) if 'dilation_offset' in metadata else None
        )
        smp.rotator_angle = (
            float(metadata.get('rotator_angle')) if 'rotator_angle' in metadata else None
        )
        smp.sample_slot = metadata.get('sample_slot')

    def _map_results(self, thermal_data) -> None:
        """Helper method to map data arrays to the results section."""
        if not self.results:
            self.results = [ThermalResult()]

        res = self.results[0]
        res.time_stamp = thermal_data.time_stamp if thermal_data.time_stamp is not None else None
        res.system_temperature = thermal_data.system_temperature if thermal_data.system_temperature is not None else None
        res.sample_temperature = thermal_data.sample_temperature if thermal_data.sample_temperature is not None else None
        res.sample_temperature_rate = thermal_data.sample_temperature_rate if thermal_data.sample_temperature_rate is not None else None
        res.sample_temperature_range = thermal_data.sample_temperature_range if thermal_data.sample_temperature_range is not None else None

        res.field = thermal_data.field * OE_TO_AM if thermal_data.field is not None else None
        res.field_rate = thermal_data.field_rate * OE_TO_AM if thermal_data.field_rate is not None else None

        res.chamber_pres = thermal_data.chamber_pres if thermal_data.chamber_pres is not None else None
        res.temperature_status = thermal_data.temperature_status if thermal_data.temperature_status is not None else None
        res.field_status = thermal_data.field_status if thermal_data.field_status is not None else None
        res.chamber_status = thermal_data.chamber_status if thermal_data.chamber_status is not None else None
        res.bridge_cycle = thermal_data.bridge_cycle if thermal_data.bridge_cycle is not None else None
        res.rotator_angle = thermal_data.rotator_angle if thermal_data.rotator_angle is not None else None
        res.therm_resistance = thermal_data.therm_resistance if thermal_data.therm_resistance is not None else None
        res.therm_resistance_rate = thermal_data.therm_resistance_rate if thermal_data.therm_resistance_rate is not None else None
        res.cell_imbalance = thermal_data.cell_imbalance if thermal_data.cell_imbalance is not None else None
        res.cell_imbalance_rate = thermal_data.cell_imbalance_rate if thermal_data.cell_imbalance_rate is not None else None
        res.tap_imbalance = thermal_data.tap_imbalance if thermal_data.tap_imbalance is not None else None
        res.coarse_dac_imbalance = thermal_data.coarse_dac_imbalance if thermal_data.coarse_dac_imbalance is not None else None
        res.fine_dac_imbalance = thermal_data.fine_dac_imbalance if thermal_data.fine_dac_imbalance is not None else None
        res.loop_imbalance = thermal_data.loop_imbalance if thermal_data.loop_imbalance is not None else None
        res.dilation = thermal_data.dilation if thermal_data.dilation is not None else None
        res.dilation_rate = thermal_data.dilation_rate if thermal_data.dilation_rate is not None else None
        res.therm_exp_coeff_raw = thermal_data.therm_exp_coeff_raw if thermal_data.therm_exp_coeff_raw is not None else None
        res.therm_exp_coeff = thermal_data.therm_exp_coeff if thermal_data.therm_exp_coeff is not None else None
        res.therm_exp_coeff_compare = thermal_data.therm_exp_coeff_compare if thermal_data.therm_exp_coeff_compare is not None else None
        res.therm_exp_coeff_diff_percentage = thermal_data.therm_exp_coeff_diff_percentage if thermal_data.therm_exp_coeff_diff_percentage is not None else None
        res.therm_exp_coeff_diff_absolute = thermal_data.therm_exp_coeff_diff_absolute if thermal_data.therm_exp_coeff_diff_absolute is not None else None
        res.therm_exp_coeff_baseline = thermal_data.therm_exp_coeff_baseline if thermal_data.therm_exp_coeff_baseline is not None else None
        res.therm_exp_coeff_reference = thermal_data.therm_exp_coeff_reference if thermal_data.therm_exp_coeff_reference is not None else None
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
    date_time = Quantity(type=str, description='Calibration Date/Time.')

class DSCInitialConditions(ArchiveSection):
    temperature = Quantity(type=str, description='Initial temperature.')
    purge_gas = Quantity(type=str, description='Purge gas used.')
    purge_gas_rate = Quantity(type=str, description='Purge gas rate.')
    baseline_filename = Quantity(type=str, description='Baseline filename.')
    end_condition = Quantity(type=str, description='End condition.')
    total_points_in_run = Quantity(type=str, description='Total points in run.')

class DSCManualTuneCalibration(ArchiveSection):
    date = Quantity(type=str)
    slope = Quantity(type=str)
    coarse_balance = Quantity(type=str)
    fine_balance = Quantity(type=str)

class DSCSmartScanCalibration(ArchiveSection):
    date = Quantity(type=str)
    smartscan_enabled = Quantity(type=str)
    calibration_file = Quantity(type=str)
    starting_temperature = Quantity(type=str)
    ending_temperature = Quantity(type=str)
    number_of_steps = Quantity(type=str)

class DSCSampleTemperatureCalibration(ArchiveSection):
    date = Quantity(type=str)
    reference = Quantity(type=str, description='Headers for reference row.')
    indium = Quantity(type=str, description='Indium calibration values.')

class DSCFurnaceTemperatureCalibration(ArchiveSection):
    minimum = Quantity(type=str, description='Minimum furnace temperature.')
    maximum = Quantity(type=str, description='Maximum furnace temperature.')

class DSCFurnaceCalibrationComputed(ArchiveSection):
    date = Quantity(type=str)
    setpoints = Quantity(type=str, description='Headers for the setpoints table.')

class DSCHeatFlowCalibrationValues(ArchiveSection):
    reference = Quantity(type=str, description='Headers for reference row.')
    indium = Quantity(type=str, description='Indium heat flow values.')

class DSCHeatFlowCalibrationComputed(ArchiveSection):
    date = Quantity(type=str)
    k_ts = Quantity(type=str, description='K(Ts) calibration polynomial (if extracted).')

class DSCProfileValues(ArchiveSection):
    software_version = Quantity(type=str)
    firmware_version = Quantity(type=str)
    instrument_serial_number = Quantity(type=str)
    load_temperature = Quantity(type=str)
    go_to_temp_rate = Quantity(type=str)
    maximum_allowed_temperature = Quantity(type=str)
    helium_purge = Quantity(type=str)
    liquid_nitrogen = Quantity(type=str)
    data_taken_using_the = Quantity(type=str)
    filter_factor = Quantity(type=str)
    cooling_device = Quantity(type=str)
    wavelet_denoising_used = Quantity(type=str)
    autoslope_used = Quantity(type=str)

class DSCResult(MeasurementResult):
    """Section for storing extracted DSC data arrays."""
    time = Quantity(type=np.float64, shape=['*'])
    unsubtracted_heat_flow = Quantity(type=np.float64, shape=['*'])
    baseline_heat_flow = Quantity(type=np.float64, shape=['*'])
    program_temperature = Quantity(type=np.float64, shape=['*'], unit='°C')
    sample_temperature = Quantity(type=np.float64, shape=['*'], unit='°C')
    approx_gas_flow = Quantity(type=np.float64, shape=['*'])
    heat_flow_calibration = Quantity(type=np.float64, shape=['*'])
    uncorrected_heat_flow = Quantity(type=np.float64, shape=['*'])


# ==========================================
# 5. MAIN DSC ENTRY DATA
# ==========================================
class DSCMeasurement(Measurement, EntryData):
    """Main EntryData schema triggered by the PerkinElmer DSC parser."""

    m_def = Section(
        a_eln=dict(lane_width='600px'),
    )

    data_file = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
        a_browser=dict(adaptor='RawFileAdaptor'),
        description='The uploaded raw data file (.txt) for the DSC measurement.'
    )

    # General Root Metadata
    sample_id_metadata = Quantity(type=str)
    serial_number = Quantity(type=str)
    operator_id = Quantity(type=str)
    sample_weight = Quantity(type=str)
    display_weight = Quantity(type=str)
    data_collected = Quantity(type=str)
    comments = Quantity(type=str)
    method_steps = Quantity(type=str, shape=['*'])

    # Validation Metadata
    validation_status = Quantity(type=str)
    validation_by = Quantity(type=str)
    validation_date = Quantity(type=str)

    # Subsections
    calibration_information = SubSection(section_def=DSCCalibrationInformation)
    initial_conditions = SubSection(section_def=DSCInitialConditions)
    manual_tune_calibration = SubSection(section_def=DSCManualTuneCalibration)
    smartscan_calibration = SubSection(section_def=DSCSmartScanCalibration)
    sample_temperature_calibration = SubSection(section_def=DSCSampleTemperatureCalibration)
    furnace_temperature_calibration = SubSection(section_def=DSCFurnaceTemperatureCalibration)
    furnace_calibration_computed = SubSection(section_def=DSCFurnaceCalibrationComputed)
    heat_flow_calibration_values = SubSection(section_def=DSCHeatFlowCalibrationValues)
    heat_flow_calibration_computed = SubSection(section_def=DSCHeatFlowCalibrationComputed)
    profile_values = SubSection(section_def=DSCProfileValues)

    results = SubSection(section_def=DSCResult, repeats=True)

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

        # 1. Map Root Metadata
        self.sample_id_metadata = dsc_data.metadata.get('Sample ID')
        self.serial_number = dsc_data.metadata.get('Serial Number')
        self.operator_id = dsc_data.metadata.get('Operator ID')
        self.sample_weight = dsc_data.metadata.get('Sample Weight')
        self.display_weight = dsc_data.metadata.get('Display Weight')
        self.data_collected = dsc_data.metadata.get('Data Collected')
        self.comments = dsc_data.metadata.get('Comment')
        self.method_steps = dsc_data.method_steps

        self.validation_status = dsc_data.metadata.get('Validation_Validated')
        self.validation_by = dsc_data.metadata.get('Validation_By')
        self.validation_date = dsc_data.metadata.get('Validation_Date')

        # Map Calibration Information (Top block)
        cal_info = DSCCalibrationInformation()
        cal_info.filename = dsc_data.metadata.get('Calibration Information_Filename')
        cal_info.date_time = dsc_data.metadata.get('Calibration Information_Date/Time')
        self.calibration_information = cal_info

        # 2. Map Initial Conditions
        ic = DSCInitialConditions()
        ic.temperature = dsc_data.metadata.get('Initial Conditions_Temperature')
        ic.purge_gas = dsc_data.metadata.get('Initial Conditions_Purge Gas')
        ic.purge_gas_rate = dsc_data.metadata.get('Initial Conditions_Purge Gas Rate')
        ic.baseline_filename = dsc_data.metadata.get('Initial Conditions_Baseline Filename')
        ic.end_condition = dsc_data.metadata.get('Initial Conditions_End Condition')
        ic.total_points_in_run = dsc_data.metadata.get('Initial Conditions_Total Points in Run')
        self.initial_conditions = ic

        # 3. Map Manual Tune Calibration
        mt = DSCManualTuneCalibration()
        mt.date = dsc_data.metadata.get('DSC8500 MANUAL TUNE CALIBRATION VALUES_Date')
        mt.slope = dsc_data.metadata.get('DSC8500 MANUAL TUNE CALIBRATION VALUES_Slope')
        mt.coarse_balance = dsc_data.metadata.get('DSC8500 MANUAL TUNE CALIBRATION VALUES_Coarse Balance')
        mt.fine_balance = dsc_data.metadata.get('DSC8500 MANUAL TUNE CALIBRATION VALUES_Fine Balance')
        self.manual_tune_calibration = mt

        # 4. Map SmartScan Calibration
        sc = DSCSmartScanCalibration()
        sc.date = dsc_data.metadata.get('DSC8500 SMARTSCAN CALIBRATION VALUES_Date')
        sc.smartscan_enabled = dsc_data.metadata.get('DSC8500 SMARTSCAN CALIBRATION VALUES_SmartScan Enabled')
        sc.calibration_file = dsc_data.metadata.get('DSC8500 SMARTSCAN CALIBRATION VALUES_Calibration File')
        sc.starting_temperature = dsc_data.metadata.get('DSC8500 SMARTSCAN CALIBRATION VALUES_Starting Temperature')
        sc.ending_temperature = dsc_data.metadata.get('DSC8500 SMARTSCAN CALIBRATION VALUES_Ending Temperature')
        sc.number_of_steps = dsc_data.metadata.get('DSC8500 SMARTSCAN CALIBRATION VALUES_Number of Steps')
        self.smartscan_calibration = sc

        # 5. Map Sample Temperature Calibration
        st_cal = DSCSampleTemperatureCalibration()
        st_cal.date = dsc_data.metadata.get('DSC8500 SAMPLE TEMPERATURE CALIBRATION VALUES_Date')
        st_cal.reference = dsc_data.metadata.get('DSC8500 SAMPLE TEMPERATURE CALIBRATION VALUES_Reference')
        st_cal.indium = dsc_data.metadata.get('DSC8500 SAMPLE TEMPERATURE CALIBRATION VALUES_Indium')
        self.sample_temperature_calibration = st_cal

        # 6. Map Furnace Temperature Calibration
        ft = DSCFurnaceTemperatureCalibration()
        ft.minimum = dsc_data.metadata.get('DSC8500 FURNACE TEMPERATURE CALIBRATION VALUES_Minimum')
        ft.maximum = dsc_data.metadata.get('DSC8500 FURNACE TEMPERATURE CALIBRATION VALUES_Maximum')
        self.furnace_temperature_calibration = ft

        # 7. Map Furnace Calibration Computed Results
        fc_comp = DSCFurnaceCalibrationComputed()
        fc_comp.date = dsc_data.metadata.get('FURNACE CALIBRATION COMPUTED RESULTS_Date')
        fc_comp.setpoints = dsc_data.metadata.get('FURNACE CALIBRATION COMPUTED RESULTS_Setpoints (°C)')
        self.furnace_calibration_computed = fc_comp

        # 8. Map Heat Flow Calibration Values
        hf_val = DSCHeatFlowCalibrationValues()
        hf_val.reference = dsc_data.metadata.get('DSC8500 HEAT FLOW CALIBRATION VALUES_Reference')
        hf_val.indium = dsc_data.metadata.get('DSC8500 HEAT FLOW CALIBRATION VALUES_Indium')
        self.heat_flow_calibration_values = hf_val

        # 9. Map Heat Flow Calibration Computed Results
        hf_comp = DSCHeatFlowCalibrationComputed()
        hf_comp.date = dsc_data.metadata.get('DSC8500 HEAT FLOW CALIBRATION COMPUTED RESULTS_Date')
        hf_comp.k_ts = dsc_data.metadata.get('DSC8500 HEAT FLOW CALIBRATION COMPUTED RESULTS_K(Ts)')
        self.heat_flow_calibration_computed = hf_comp

        # 10. Map Profile Values
        pv = DSCProfileValues()
        pv.software_version = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Software Version')
        pv.firmware_version = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Firmware Version')
        pv.instrument_serial_number = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Instrument Serial Number')
        pv.load_temperature = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Load Temperature')
        pv.go_to_temp_rate = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Go To Temp Rate')
        pv.maximum_allowed_temperature = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Maximum Allowed Temperature')
        pv.helium_purge = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Helium Purge')
        pv.liquid_nitrogen = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Liquid Nitrogen')
        pv.data_taken_using_the = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Data taken using the')
        pv.filter_factor = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Filter Factor')
        pv.cooling_device = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Cooling Device')
        pv.wavelet_denoising_used = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Wavelet Denoising used')
        pv.autoslope_used = dsc_data.metadata.get('PROFILE VALUES FOR THIS DATA_Autoslope Used')
        self.profile_values = pv

        # 11. Map Results Arrays
        if not self.results:
            self.results = [DSCResult()]

        res = self.results[0]
        res.time = dsc_data.time
        res.unsubtracted_heat_flow = dsc_data.unsubtracted_heat_flow
        res.baseline_heat_flow = dsc_data.baseline_heat_flow
        res.program_temperature = dsc_data.program_temperature
        res.sample_temperature = dsc_data.sample_temperature
        res.approx_gas_flow = dsc_data.approx_gas_flow
        res.heat_flow_calibration = dsc_data.calibration
        res.uncorrected_heat_flow = dsc_data.heat_flow

m_package.__init_metainfo__()