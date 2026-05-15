from unittest.mock import MagicMock, patch

from nomad.datamodel import EntryArchive

from nomad_measurements_thermal.parsers.parser import ThermalParser


def test_is_mainfile():
    """Test the parser's gatekeeper logic."""
    parser = ThermalParser()

    # Valid thermal file structure
    valid_content = (
        '[Header]\n'
        'BEGIN:PARAMS\n'
        'cell_constant,1.0\n'
        '[Data]\n'
        'TimeStamp (sec)...'
    )
    assert parser.is_mainfile(
        'test_file.dat', 'text/plain', valid_content.encode(), valid_content
    )

    # Invalid file structure
    invalid_content = 'Just some random text without the proper headers.'
    assert not parser.is_mainfile(
        'test_file.dat', 'text/plain', invalid_content.encode(), invalid_content
    )


@patch(
    'nomad_measurements_thermal.schema_packages.schema_package.ThermalMeasurement.normalize'
)
def test_parse(mock_normalize):
    """Test that the parser builds the schema and triggers normalize."""
    parser = ThermalParser()
    archive = EntryArchive()

    # FIX: Mock the context so the parser can access the logger
    archive.m_context = MagicMock()

    # Run the parse function
    parser.parse('path/to/my_test_file.dat', archive, None)

    # Verify the EntryArchive was populated
    assert archive.data is not None
    assert archive.data.data_file == 'my_test_file.dat'

    # Verify the normalize function was called exactly once
    mock_normalize.assert_called_once()
