from nomad.datamodel.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser

# Import the specialized schemas
from nomad_measurements_thermal.schema_packages.schema_package import (
    DSCMeasurement,
    TADSCMeasurement,
    ThermalMeasurement,
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
        has_dilatometry_sections = '[Header]' in text and '[Data]' in text
        has_dilatometry_marker = any(
            marker in text
            for marker in (
                'BEGIN:PARAMS',
                'dilation_offset',
                'cell_constant',
                'Therm Resistance',
                'Dilation (ppm)',
            )
        )
        if has_dilatometry_sections and has_dilatometry_marker:
            return True

        # 2. PerkinElmer DSC Check
        has_dsc_markers = (
            'Sample Weight:' in text and 'Method Steps:' in text and 'Heat Flow' in text
        )
        if has_dsc_markers:
            return True

        # 3. TA Instruments DSC Check
        has_ta_dsc_markers = 'CLOSED' in text and 'Instrument' in text
        if has_ta_dsc_markers:
            return True

        return False

    def parse(
        self,
        mainfile: str,
        archive: EntryArchive,
        logger=None,
        child_archives=None,
    ) -> None:
        logger = logger or archive.m_context.logger

        filename = mainfile.rsplit('/', maxsplit=1)[-1]

        # Read file as raw binary bytes to bypass character encoding traps smoothly
        with archive.m_context.raw_file(filename, 'rb') as f:
            raw_bytes = f.read(4000)

        content_peek = raw_bytes.decode('utf-8', errors='ignore')
        if '\x00' in content_peek:
            content_peek = raw_bytes.decode('utf-16', errors='ignore')

        # Route matching signatures strictly to their corresponding schema classes
        if '[Header]' in content_peek and '[Data]' in content_peek:
            logger.info('Routing to Dilatometry schema.')
            entry = ThermalMeasurement()
        elif 'Method Steps:' in content_peek and 'Sample Weight:' in content_peek:
            logger.info('Routing to PerkinElmer DSC schema.')
            entry = DSCMeasurement()
        elif 'CLOSED' in content_peek and 'Instrument' in content_peek:
            logger.info('Routing to TA Instruments DSC schema.')
            entry = TADSCMeasurement()
        else:
            logger.error(f'Unrecognized thermal file format: {filename}')
            return

        entry.data_file = filename
        archive.data = entry

        entry.normalize(archive, logger)
