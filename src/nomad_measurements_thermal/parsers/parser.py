from nomad.datamodel.context import ServerContext
from nomad.datamodel.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser
from nomad_measurements.utils import create_archive

# Import the specialized schemas AND the new wrapper
from nomad_measurements_thermal.schema_packages.schema_package import (
    ARCMeasurement,
    DilatometryMeasurement,
    DSCMeasurement,
    RawFileThermalData,
    TADSCMeasurement,
)


class ThermalParser(MatchingParser):
    def is_mainfile(
        self,
        filename: str,
        mime: str,
        buffer: bytes,
        decoded_buffer: str,
        compression: str = None,
    ) -> bool:
        """Gatekeeper for Dilatometry, PerkinElmer DSC, and TA DSC files."""
        if not super().is_mainfile(filename, mime, buffer, decoded_buffer, compression):
            return False

        text = decoded_buffer if decoded_buffer else ''
        if not text and buffer:
            text = buffer.decode('utf-8', errors='ignore')
            if '\x00' in text:
                text = buffer.decode('utf-16', errors='ignore')

        if not text:
            return False

        # 1. Dilatometry Check
        is_dilatometry = (
            '[Header]' in text
            and '[Data]' in text
            and any(
                marker in text
                for marker in (
                    'BEGIN:PARAMS',
                    'dilation_offset',
                    'cell_constant',
                    'Therm Resistance',
                    'Dilation (ppm)',
                )
            )
        )

        # 2. PerkinElmer DSC Check
        is_pe_dsc = (
            'Sample Weight:' in text and 'Method Steps:' in text and 'Heat Flow' in text
        )

        # 3. TA Instruments DSC Check
        is_ta_dsc = 'CLOSED' in text and 'Instrument' in text

        # 4. ARC Semicolon Format Check
        is_arc = (
            'Test Cell Type;' in text
            and 'Serial Number;Current Time;Sample Temperature' in text
        )

        return is_dilatometry or is_pe_dsc or is_ta_dsc or is_arc

    def parse(
        self,
        mainfile: str,
        archive: EntryArchive,
        logger=None,
        child_archives=None,
    ) -> None:
        logger = logger or archive.m_context.logger

        # Extract the filename, handling server context paths correctly
        data_file = mainfile.rsplit('/', maxsplit=1)[-1]
        if isinstance(archive.m_context, ServerContext):
            data_file = mainfile.split('/raw/', 1)[1]

        # Read file as raw binary bytes to bypass character encoding traps smoothly
        with archive.m_context.raw_file(data_file, 'rb') as f:
            raw_bytes = f.read(4000)

        content_peek = raw_bytes.decode('utf-8', errors='ignore')
        if '\x00' in content_peek:
            content_peek = raw_bytes.decode('utf-16', errors='ignore')

        # Route matching signatures strictly to their corresponding schema classes
        if '[Header]' in content_peek and '[Data]' in content_peek:
            logger.info('Routing to Dilatometry schema.')
            entry = DilatometryMeasurement()
        elif 'Method Steps:' in content_peek and 'Sample Weight:' in content_peek:
            logger.info('Routing to PerkinElmer DSC schema.')
            entry = DSCMeasurement()
        elif 'CLOSED' in content_peek and 'Instrument' in content_peek:
            logger.info('Routing to TA Instruments DSC schema.')
            entry = TADSCMeasurement()
        elif 'Test Cell Type;' in content_peek and 'Sample Temperature' in content_peek:
            logger.info('Routing to ARC schema.')
            entry = ARCMeasurement()
        else:
            logger.error(f'Unrecognized thermal file format: {data_file}')
            return

        # Assign the file name to the entry
        entry.data_file = data_file

        # Create the separate editable .archive.json file to preserve ELN edits
        archive_name = f'{"".join(data_file.split(".")[:-1])}.archive.json'

        # Link the raw file to the generated ELN using the placeholder
        archive.data = RawFileThermalData(
            measurement=create_archive(entry, archive, archive_name)
        )

        # Clean up the display name in the GUI
        archive.metadata.entry_name = f'{data_file} data file'
