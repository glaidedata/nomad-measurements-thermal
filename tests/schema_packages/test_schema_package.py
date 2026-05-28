from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from nomad.datamodel import EntryArchive, EntryMetadata

# Import the data models directly from your reader package
from readers_ientrance import DSCData, TADSCData, ThermalData

from nomad_measurements_thermal.schema_packages.schema_package import (
    DSCMeasurement,
    TADSCMeasurement,
    ThermalMeasurement,
)


@pytest.fixture
def mock_thermal_data():
    """Mock output simulating Dilatometry reader."""
    metadata = {
        'TITLE': 'Test Mock Title',
        'sample_length': '2.5',
        'cell_constant': '0.15',
        'offset_mode': 'set',
    }
    return ThermalData(
        metadata=metadata,
        time_stamp=np.array([1.0, 2.0]),
        system_temperature=np.array([300.0, 305.0]),
        field=np.array([100.0, 200.0]),
        dilation=np.array([0.5, 0.6]),
        comment=['Start', 'End'],
    )


@pytest.fixture
def mock_dsc_data():
    """Mock output simulating PerkinElmer reader."""
    mock_data = MagicMock(spec=DSCData)
    mock_data.metadata = {
        'Sample ID': 'Test_DNA',
        'Sample Weight': '1.080 mg',
        'Data Collected': '27/10/2025 12:17:34',
    }
    mock_data.method_steps = ['Step 1', 'Step 2']
    mock_data.tables = {}

    # All arrays must be present to satisfy the schema mappings
    mock_data.time = np.array([0.0, 0.5])
    mock_data.unsubtracted_heat_flow = np.array([-30.0, -30.0])
    mock_data.baseline_heat_flow = np.array([0.0, 0.0])
    mock_data.program_temperature = np.array([165.0, 165.0])
    mock_data.sample_temperature = np.array([164.9, 165.1])
    mock_data.approx_gas_flow = np.array([20.0, 20.0])
    mock_data.heat_flow_calibration = np.array([1.36, 1.36])
    mock_data.uncorrected_heat_flow = np.array([-30.0, -30.0])

    return mock_data


@pytest.fixture
def mock_ta_dsc_data():
    """Mock output simulating TA Instruments reader."""
    metadata = {
        'Sample': 'Zeolite',
        'Size': '28.0000 mg',
        'Date': '2026-03-09',
        'Time': '10:48:37',
        'Instrument': 'DSC Q2000',
        'Kcell': '1.00000',
        'InstCalFile': 'Tzero: tzero.TZR | Baseline: base.078 | Sapphire: sap.078',
    }
    return TADSCData(
        metadata=metadata,
        method_steps=['1: Ramp', '2: Isothermal'],
        time=np.array([1.0, 2.0]),
        sample_temperature=np.array([25.0, 30.0]),
        heat_flow=np.array([-5.0, -4.5]),
        heat_capacity=np.array([200.0, 210.0]),
        approx_gas_flow=np.array([50.0, 50.0]),
    )


@patch('nomad_measurements_thermal.schema_packages.schema_package.read_thermal_dat')
def test_thermal_normalize(mock_read_thermal_dat, mock_thermal_data):
    """Verify Dilatometry normalization."""
    mock_read_thermal_dat.return_value = mock_thermal_data

    archive = EntryArchive()
    archive.m_context = MagicMock()
    archive.metadata = EntryMetadata(entry_name='dummy')

    mock_file = MagicMock()
    mock_file.name = 'dummy_path.dat'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    entry = ThermalMeasurement(data_file='dummy.dat')
    entry.normalize(archive, MagicMock())

    assert entry.title == 'Test Mock Title'
    assert entry.sample[0].sample_length == 2.5  # noqa: PLR2004
    assert np.array_equal(entry.results[0].time_stamp.magnitude, [1.0, 2.0])


@patch('nomad_measurements_thermal.schema_packages.schema_package.read_perkinelmer_dsc')
def test_dsc_normalize(mock_read_perkinelmer_dsc, mock_dsc_data):
    """Verify PerkinElmer DSC normalization."""
    mock_read_perkinelmer_dsc.return_value = mock_dsc_data

    archive = EntryArchive()
    archive.m_context = MagicMock()
    archive.metadata = EntryMetadata(entry_name='dummy')

    mock_file = MagicMock()
    mock_file.name = 'dummy_path.txt'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    entry = DSCMeasurement(data_file='dummy.txt')
    entry.normalize(archive, MagicMock())

    assert entry.sample_id == 'Test_DNA'
    assert entry.sample_weight.magnitude == 1.08  # noqa: PLR2004
    assert np.array_equal(entry.results[0].unsubtracted_heat_flow, [-30.0, -30.0])


@patch('nomad_measurements_thermal.schema_packages.schema_package.read_ta_dsc')
def test_ta_dsc_normalize(mock_read_ta_dsc, mock_ta_dsc_data):
    """Verify TA Instruments DSC normalization."""
    mock_read_ta_dsc.return_value = mock_ta_dsc_data

    archive = EntryArchive()
    archive.m_context = MagicMock()
    archive.metadata = EntryMetadata(entry_name='dummy')

    mock_file = MagicMock()
    mock_file.name = 'dummy_path_ta.txt'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    entry = TADSCMeasurement(data_file='dummy_ta.txt')
    entry.normalize(archive, MagicMock())

    # Check mapping logic
    assert entry.sample_id == 'Zeolite'
    assert entry.sample_weight.magnitude == 28.0  # noqa: PLR2004
    assert entry.data_collected == '2026-03-09 10:48:37'
    assert entry.instrument == 'DSC Q2000'
    assert entry.cell_constant == 1.0  # noqa: PLR2004

    # Check regex extraction for calibration strings
    assert entry.calibration_file_tzero == 'tzero.TZR'
    assert entry.calibration_file_baseline == 'base.078'
    assert entry.calibration_file_sapphire == 'sap.078'

    # Check results mappings
    assert np.array_equal(entry.results[0].heat_capacity.magnitude, [200.0, 210.0])
