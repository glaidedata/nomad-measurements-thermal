from unittest.mock import MagicMock, patch

from nomad.datamodel import EntryArchive

from nomad_measurements_thermal.parsers.parser import ThermalParser


def test_is_mainfile():
    """Test the parser's gatekeeper logic for all four file formats."""
    parser = ThermalParser()

    # 1. Valid Dilatometry file structure
    valid_dilatometry = "[Header]\nBEGIN:PARAMS\ncell_constant,1.0\n[Data]\nTimeStamp"
    assert parser.is_mainfile(
        "test_file.dat", "text/plain", valid_dilatometry.encode(), valid_dilatometry
    )

    # 2. Valid PerkinElmer DSC file structure
    valid_dsc = "Sample Weight: 1.080 mg\nMethod Steps:\nHeat Flow\n"
    assert parser.is_mainfile(
        "test_file.txt", "text/plain", valid_dsc.encode(), valid_dsc
    )

    # 3. Valid TA Instruments DSC file
    valid_ta = "CLOSED\nLanguage\nInstrument Q2000\nDSC\nStartOfData\nOrgMethod\n"
    assert parser.is_mainfile(
        "test_ta.txt", "text/plain", valid_ta.encode(), valid_ta
    )

    # 4. Valid ARC file structure
    valid_arc = "Test Cell Type;SS316L\nSerial Number;Current Time;Sample Temperature\n"
    assert parser.is_mainfile(
        "test_arc.txt", "text/plain", valid_arc.encode(), valid_arc
    )

    # 5. Invalid file structure
    invalid_content = "Just some random text without the proper headers."
    assert not parser.is_mainfile(
        "test_invalid.dat", "text/plain", invalid_content.encode(), invalid_content
    )


@patch(
    "nomad_measurements_thermal.schema_packages."
    "schema_package.DilatometryMeasurement.normalize"
)
def test_parse_thermal(mock_normalize):
    """Verify routing to the Dilatometry schema."""
    parser = ThermalParser()
    archive = EntryArchive()
    archive.m_context = MagicMock()

    mock_file = MagicMock()
    mock_file.read.return_value = b"[Header]\nBEGIN:PARAMS\n[Data]\n"
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    parser.parse("path/to/my_test_file.dat", archive, None)

    assert archive.data is not None
    assert archive.data.__class__.__name__ == "DilatometryMeasurement"
    mock_normalize.assert_called_once()


@patch(
    "nomad_measurements_thermal.schema_packages."
    "schema_package.DSCMeasurement.normalize"
)
def test_parse_pe_dsc(mock_normalize):
    """Verify routing to the PerkinElmer DSC schema."""
    parser = ThermalParser()
    archive = EntryArchive()
    archive.m_context = MagicMock()

    mock_file = MagicMock()
    mock_file.read.return_value = b"Sample Weight: 1.080\nMethod Steps:\n"
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    parser.parse("path/to/my_test_file.txt", archive, None)

    assert archive.data is not None
    assert archive.data.__class__.__name__ == "DSCMeasurement"
    mock_normalize.assert_called_once()


@patch(
    "nomad_measurements_thermal.schema_packages."
    "schema_package.TADSCMeasurement.normalize"
)
def test_parse_ta_dsc(mock_normalize):
    """Verify routing to the TA Instruments DSC schema."""
    parser = ThermalParser()
    archive = EntryArchive()
    archive.m_context = MagicMock()

    mock_file = MagicMock()
    mock_file.read.return_value = b"CLOSED\nInstrument\nDSC\n"
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    parser.parse("path/to/ta_test_file.txt", archive, None)

    assert archive.data is not None
    assert archive.data.__class__.__name__ == "TADSCMeasurement"
    mock_normalize.assert_called_once()


@patch(
    "nomad_measurements_thermal.schema_packages."
    "schema_package.ARCMeasurement.normalize"
)
def test_parse_arc(mock_normalize):
    """Verify routing to the ARC schema."""
    parser = ThermalParser()
    archive = EntryArchive()
    archive.m_context = MagicMock()

    mock_file = MagicMock()
    mock_file.read.return_value = b"Test Cell Type;\nSample Temperature\n"
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file

    parser.parse("path/to/arc_test_file.txt", archive, None)

    assert archive.data is not None
    assert archive.data.__class__.__name__ == "ARCMeasurement"
    mock_normalize.assert_called_once()