from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from nomad.datamodel import EntryArchive, EntryMetadata
from readers_ientrance.thermal_reader import ThermalData

from nomad_measurements_thermal.schema_packages.schema_package import (
    ThermalMeasurement,
)


@pytest.fixture
def mock_thermal_data():
    """Returns a mock ThermalData object simulating the output of your reader."""
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


@patch('nomad_measurements_thermal.schema_packages.schema_package.read_thermal_dat')
def test_thermal_measurement_normalize(mock_read_thermal_dat, mock_thermal_data):
    """Tests that the normalize function correctly maps data and converts units."""
    # 1. Instruct the mock reader to return our mock_thermal_data
    mock_read_thermal_dat.return_value = mock_thermal_data

    # 2. Setup a dummy NOMAD archive and mock the raw_file context manager
    archive = EntryArchive()
    archive.m_context = MagicMock()

    # FIX: Use actual EntryMetadata to prevent the base class warning
    archive.metadata = EntryMetadata(entry_name='dummy_entry_name')

    mock_file = MagicMock()
    mock_file.name = 'dummy_absolute_path.dat'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    # 3. Instantiate the schema and trigger normalization
    entry = ThermalMeasurement(data_file='dummy_path.dat')
    entry.normalize(archive, MagicMock())

    # --- VERIFICATIONS ---

    # Verify Top-Level Metadata
    assert entry.title == 'Test Mock Title'

    # Verify Sample Section
    assert len(entry.sample) == 1
    smp = entry.sample[0]
    assert smp.sample_length == 2.5  # noqa: PLR2004
    assert smp.cell_constant == 0.15  # noqa: PLR2004
    assert smp.offset_mode == 'set'
    assert smp.dilation_offset == 0.01  # noqa: PLR2004
    assert smp.rotator_angle == 45.0  # noqa: PLR2004
    assert smp.sample_slot == 'A1'

    # Verify Results Section
    assert len(entry.results) == 1
    res = entry.results[0]

    # FIX: Add .magnitude to extract the raw numpy array from the pint.Quantity
    assert np.array_equal(res.time_stamp.magnitude, [1.0, 2.0])
    assert np.array_equal(res.system_temperature.magnitude, [300.0, 305.0])
    assert np.array_equal(res.dilation.magnitude, [0.5, 0.6])
    assert res.comment == [
        'Start',
        'End',
    ]  # Strings don't have units, so this stays the same

    # Verify Unit Conversion (Oe to A/m)
    OE_TO_AM = 1000.0 / (4.0 * np.pi)
    expected_field = np.array([100.0, 200.0]) * OE_TO_AM
    assert np.allclose(res.field.magnitude, expected_field)
