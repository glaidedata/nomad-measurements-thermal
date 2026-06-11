from unittest.mock import MagicMock, patch

from nomad.datamodel.datamodel import EntryArchive, EntryMetadata

from nomad_measurements_thermal.parsers.parser import ThermalParser
from nomad_measurements_thermal.schema_packages.schema_package import (
    ARCMeasurement,
    DilatometryMeasurement,
    DSCMeasurement,
    RawFileThermalData,
    TADSCMeasurement,
)


def test_is_mainfile():
    """Test the parser's gatekeeper logic for all four file formats."""
    parser = ThermalParser()

    # 1. Valid Dilatometry file structure
    valid_dilatometry = '[Header]\nBEGIN:PARAMS\ncell_constant,1.0\n[Data]\nTimeStamp'
    assert parser.is_mainfile(
        'test_file.dat', 'text/plain', valid_dilatometry.encode(), valid_dilatometry
    )

    # 2. Valid PerkinElmer DSC file structure
    valid_dsc = 'Sample Weight: 1.080 mg\nMethod Steps:\nHeat Flow\n'
    assert parser.is_mainfile(
        'test_file.txt', 'text/plain', valid_dsc.encode(), valid_dsc
    )

    # 3. Valid TA Instruments DSC file
    valid_ta = 'CLOSED\nLanguage\nInstrument Q2000\nDSC\nStartOfData\nOrgMethod\n'
    assert parser.is_mainfile('test_ta.txt', 'text/plain', valid_ta.encode(), valid_ta)

    # 4. Valid ARC file structure
    valid_arc = 'Test Cell Type;SS316L\nSerial Number;Current Time;Sample Temperature\n'
    assert parser.is_mainfile(
        'test_arc.txt', 'text/plain', valid_arc.encode(), valid_arc
    )

    # 5. Invalid file structure
    invalid_content = 'Just some random text without the proper headers.'
    assert not parser.is_mainfile(
        'test_invalid.dat', 'text/plain', invalid_content.encode(), invalid_content
    )


@patch('nomad_measurements_thermal.parsers.parser.create_archive')
def test_parse_thermal(mock_create_archive):
    """Verify routing to the Dilatometry schema via Two-Archive."""
    mock_create_archive.return_value = 'mocked_archive_reference'
    parser = ThermalParser()

    archive = EntryArchive()
    archive.metadata = EntryMetadata()
    archive.m_context = MagicMock()

    mock_file = MagicMock()
    mock_file.read.return_value = b'[Header]\nBEGIN:PARAMS\n[Data]\n'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    parser.parse('path/to/my_test_file.dat', archive, None)

    # Check that the placeholder was attached
    assert isinstance(archive.data, RawFileThermalData)
    assert archive.data.measurement.m_proxy_value == 'mocked_archive_reference'

    # Check that the correct ELN was created
    mock_create_archive.assert_called_once()
    entry, _, archive_name = mock_create_archive.call_args[0]
    assert isinstance(entry, DilatometryMeasurement)
    assert entry.data_file == 'my_test_file.dat'
    assert archive_name == 'my_test_file.archive.json'


@patch('nomad_measurements_thermal.parsers.parser.create_archive')
def test_parse_pe_dsc(mock_create_archive):
    """Verify routing to the PerkinElmer DSC schema via Two-Archive."""
    mock_create_archive.return_value = 'mocked_archive_reference'
    parser = ThermalParser()

    archive = EntryArchive()
    archive.metadata = EntryMetadata()
    archive.m_context = MagicMock()

    mock_file = MagicMock()
    mock_file.read.return_value = b'Sample Weight: 1.080\nMethod Steps:\n'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    parser.parse('path/to/my_test_file.txt', archive, None)

    assert isinstance(archive.data, RawFileThermalData)
    assert archive.data.measurement.m_proxy_value == 'mocked_archive_reference'

    mock_create_archive.assert_called_once()
    entry, _, archive_name = mock_create_archive.call_args[0]
    assert isinstance(entry, DSCMeasurement)
    assert entry.data_file == 'my_test_file.txt'
    assert archive_name == 'my_test_file.archive.json'


@patch('nomad_measurements_thermal.parsers.parser.create_archive')
def test_parse_ta_dsc(mock_create_archive):
    """Verify routing to the TA Instruments DSC schema via Two-Archive."""
    mock_create_archive.return_value = 'mocked_archive_reference'
    parser = ThermalParser()

    archive = EntryArchive()
    archive.metadata = EntryMetadata()
    archive.m_context = MagicMock()

    mock_file = MagicMock()
    mock_file.read.return_value = b'CLOSED\nInstrument\nDSC\n'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    parser.parse('path/to/ta_test_file.txt', archive, None)

    assert isinstance(archive.data, RawFileThermalData)
    assert archive.data.measurement.m_proxy_value == 'mocked_archive_reference'

    mock_create_archive.assert_called_once()
    entry, _, archive_name = mock_create_archive.call_args[0]
    assert isinstance(entry, TADSCMeasurement)
    assert entry.data_file == 'ta_test_file.txt'
    assert archive_name == 'ta_test_file.archive.json'


@patch('nomad_measurements_thermal.parsers.parser.create_archive')
def test_parse_arc(mock_create_archive):
    """Verify routing to the ARC schema via Two-Archive."""
    mock_create_archive.return_value = 'mocked_archive_reference'
    parser = ThermalParser()

    archive = EntryArchive()
    archive.metadata = EntryMetadata()
    archive.m_context = MagicMock()

    mock_file = MagicMock()
    mock_file.read.return_value = b'Test Cell Type;\nSample Temperature\n'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    parser.parse('path/to/arc_test_file.txt', archive, None)

    assert isinstance(archive.data, RawFileThermalData)
    assert archive.data.measurement.m_proxy_value == 'mocked_archive_reference'

    mock_create_archive.assert_called_once()
    entry, _, archive_name = mock_create_archive.call_args[0]
    assert isinstance(entry, ARCMeasurement)
    assert entry.data_file == 'arc_test_file.txt'
    assert archive_name == 'arc_test_file.archive.json'
