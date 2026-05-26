from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from nomad.datamodel import EntryArchive, EntryMetadata
from readers_ientrance.thermal_reader import ThermalData

# Import the newly refactored schemas
from nomad_measurements_thermal.schema_packages.schema_package import (
    DilatometryMeasurement,
    PerkinElmerDSCMeasurement,
)


@pytest.fixture
def mock_thermal_data():
    """Returns a mock ThermalData object simulating the output of the dilatometry reader."""
    metadata = {
        'TITLE': 'Test Mock Title',
        'sample_length': '2.5',
        'cell_constant': '0.15',
        'offset_mode': 'set',
        'dilation_offset': '0.01',
        'rotator_angle': '45.0',
        'sample_slot': 'A1',
    }
    return ThermalData(
        metadata=metadata,
        time_stamp=np.array([1.0, 2.0]),
        system_temperature=np.array([300.0, 305.0]),
        field=np.array([100.0, 200.0]),  # In Oersted (Oe)
        dilation=np.array([0.5, 0.6]),
        comment=['Start', 'End'],
    )


@pytest.fixture
def mock_dsc_data():
    """Returns a generic mock object simulating the output of the PerkinElmer DSC reader."""
    mock_data = MagicMock()
    mock_data.metadata = {
        'Sample ID': 'Test_DNA',
        'Serial Number': 'SN12345',
        'Operator ID': 'AG',
        'Sample Weight': '1.080 mg',
        'Display Weight': '1.080',
        'Data Collected': '27/10/2025 12:17:34',
        'Comment': 'm(S+PAN)= 616.98 mg',
        'Validation_Validated': 'No',
        'Initial Conditions_Temperature': '165.00 °C',
        'Initial Conditions_Purge Gas': 'Nitrogen',
    }
    mock_data.method_steps = ['Step 1', 'Step 2']
    mock_data.tables = {
        'SAMPLE TEMPERATURE CALIBRATION VALUES': [
            [
                'Reference',
                'Expected (°C)',
                'Measured (°C)',
                'Weight (mg)',
                'Scan rate (°C/min)',
            ],
            ['Indium', '156.600', '161.940', '9.600', '2.5'],
        ]
    }
    mock_data.time = np.array([0.0, 0.5])
    mock_data.program_temperature = np.array([165.0, 165.0])
    mock_data.sample_temperature = np.array([159.4, 159.5])
    mock_data.unsubtracted_heat_flow = np.array([-30.7, -30.7])
    mock_data.baseline_heat_flow = np.array([0.0, 0.0])
    mock_data.approx_gas_flow = np.array([20.0, 20.0])
    mock_data.heat_flow_calibration = np.array([1.36, 1.36])
    mock_data.uncorrected_heat_flow = np.array([-30.7, -30.7])

    return mock_data


@patch('nomad_measurements_thermal.schema_packages.schema_package.read_thermal_dat')
def test_dilatometry_measurement_normalize(mock_read_thermal_dat, mock_thermal_data):
    """Tests that the Dilatometry normalize function correctly maps data and converts units."""
    mock_read_thermal_dat.return_value = mock_thermal_data

    archive = EntryArchive()
    archive.m_context = MagicMock()
    archive.metadata = EntryMetadata(entry_name='dummy_entry_name')

    mock_file = MagicMock()
    mock_file.name = 'dummy_absolute_path.dat'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    entry = DilatometryMeasurement(data_file='dummy_path.dat')
    entry.normalize(archive, MagicMock())

    assert entry.title == 'Test Mock Title'

    assert len(entry.sample) == 1
    smp = entry.sample[0]
    assert smp.sample_length == 2.5  # noqa: PLR2004
    assert smp.cell_constant == 0.15  # noqa: PLR2004
    assert smp.offset_mode == 'set'

    assert len(entry.results) == 1
    res = entry.results[0]
    assert np.array_equal(res.time_stamp.magnitude, [1.0, 2.0])
    assert np.array_equal(res.system_temperature.magnitude, [300.0, 305.0])
    assert np.array_equal(res.dilation.magnitude, [0.5, 0.6])

    OE_TO_AM = 1000.0 / (4.0 * np.pi)
    expected_field = np.array([100.0, 200.0]) * OE_TO_AM
    assert np.allclose(res.field.magnitude, expected_field)


@patch('nomad_measurements_thermal.schema_packages.schema_package.read_perkinelmer_dsc')
def test_dsc_measurement_normalize(mock_read_perkinelmer_dsc, mock_dsc_data):
    """Tests that the PerkinElmer DSC normalize function correctly maps data and extracts floats."""
    mock_read_perkinelmer_dsc.return_value = mock_dsc_data

    archive = EntryArchive()
    archive.m_context = MagicMock()
    archive.metadata = EntryMetadata(entry_name='dummy_entry_name')

    mock_file = MagicMock()
    mock_file.name = 'dummy_absolute_path.txt'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    entry = PerkinElmerDSCMeasurement(data_file='dummy_path.txt')
    entry.normalize(archive, MagicMock())

    # Verify Universal Base Properties
    assert entry.sample_id == 'Test_DNA'
    assert entry.operator_id == 'AG'
    assert entry.sample_weight.magnitude == 1.080  # noqa: PLR2004
    assert entry.data_collected == '27/10/2025 12:17:34'
    assert entry.comments == 'm(S+PAN)= 616.98 mg'

    # Verify Description and Specific Properties
    assert entry.serial_number == 'SN12345'
    assert 'Method Steps:' in entry.description
    assert 'Step 1' in entry.description

    # Verify Subsection Extraction
    assert entry.initial_conditions.temperature.magnitude == 165.0  # noqa: PLR2004
    assert entry.initial_conditions.purge_gas == 'Nitrogen'

    # Verify Table Extraction
    st_cal = entry.sample_temperature_calibration
    assert st_cal.reference == ['Indium']
    assert st_cal.expected_temperature.magnitude[0] == 156.6  # noqa: PLR2004
    assert st_cal.measured_temperature.magnitude[0] == 161.94  # noqa: PLR2004

    # Verify Results Array
    assert len(entry.results) == 1
    res = entry.results[0]
    assert np.array_equal(res.time, [0.0, 0.5])
    assert np.array_equal(res.program_temperature, [165.0, 165.0])
    assert np.array_equal(res.heat_flow_calibration, [1.36, 1.36])
