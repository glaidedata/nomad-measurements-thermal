from unittest.mock import MagicMock, patch

from nomad.datamodel import EntryArchive

from nomad_measurements_thermal.parsers.parser import ThermalParser


def test_is_mainfile():
    """Test the parser's gatekeeper logic for both file formats."""
    parser = ThermalParser()

    # 1. Valid Dilatometry file structure
    valid_dilatometry = (
        '[Header]\nBEGIN:PARAMS\ncell_constant,1.0\n[Data]\nTimeStamp (sec)...'
    )
    assert parser.is_mainfile(
        'test_file.dat', 'text/plain', valid_dilatometry.encode(), valid_dilatometry
    )

    # 2. Valid DSC file structure
    valid_dsc = 'Sample Weight: 1.080 mg\nMethod Steps:\nHeat Flow\n'
    assert parser.is_mainfile(
        'test_file.txt', 'text/plain', valid_dsc.encode(), valid_dsc
    )

    # 3. Invalid file structure
    invalid_content = 'Just some random text without the proper headers.'
    assert not parser.is_mainfile(
        'test_file.dat', 'text/plain', invalid_content.encode(), invalid_content
    )


@patch(
    'nomad_measurements_thermal.schema_packages.schema_package.DilatometryMeasurement.normalize'
)
def test_parse_dilatometry(mock_normalize):
    """Test that the parser builds the Dilatometry schema and triggers normalize."""
    parser = ThermalParser()
    archive = EntryArchive()
    archive.m_context = MagicMock()

    # Mock the file read so the parser's content_peek detects Dilatometry
    mock_file = MagicMock()
    mock_file.read.return_value = '[Header]\nBEGIN:PARAMS\n[Data]\n'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    parser.parse('path/to/my_test_file.dat', archive, None)

    assert archive.data is not None
    assert archive.data.data_file == 'my_test_file.dat'
    assert archive.data.__class__.__name__ == 'DilatometryMeasurement'
    mock_normalize.assert_called_once()


@patch(
    'nomad_measurements_thermal.schema_packages.schema_package.PerkinElmerDSCMeasurement.normalize'
)
def test_parse_dsc(mock_normalize):
    """Test that the parser builds the DSC schema and triggers normalize."""
    parser = ThermalParser()
    archive = EntryArchive()
    archive.m_context = MagicMock()

    # Mock the file read so the parser's content_peek detects DSC
    mock_file = MagicMock()
    mock_file.read.return_value = 'Sample Weight: 1.080\nMethod Steps:\n'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    parser.parse('path/to/my_test_file.txt', archive, None)

    assert archive.data is not None
    assert archive.data.data_file == 'my_test_file.txt'
    assert archive.data.__class__.__name__ == 'PerkinElmerDSCMeasurement'
    mock_normalize.assert_called_once()
