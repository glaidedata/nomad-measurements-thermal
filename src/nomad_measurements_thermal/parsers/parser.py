from nomad.datamodel.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser

# Import both specialized schemas
from nomad_measurements_thermal.schema_packages.schema_package import (
    ThermalMeasurement,
    DSCMeasurement,
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
        """Gatekeeper for Dilatometry and PerkinElmer DSC files."""
        if not super().is_mainfile(filename, mime, buffer, decoded_buffer, compression):
            return False

        if not decoded_buffer:
            return False

        # 1. Dilatometry Check
        has_dilatometry_sections = '[Header]' in decoded_buffer and '[Data]' in decoded_buffer
        has_dilatometry_marker = any(
            marker in decoded_buffer
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
            'Sample Weight:' in decoded_buffer and
            'Method Steps:' in decoded_buffer and
            'Heat Flow' in decoded_buffer
        )
        if has_dsc_markers:
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

        # Extract just the filename from the path
        filename = mainfile.rsplit('/', maxsplit=1)[-1]

        # Peek into the file content to route to the correct Schema
        with archive.m_context.raw_file(filename, 'r') as f:
            content_peek = f.read(2000)

        # Route based on content signatures
        if '[Header]' in content_peek and '[Data]' in content_peek:
            logger.info('Routing to Dilatometry schema.')
            entry = ThermalMeasurement()
        elif 'Method Steps:' in content_peek and 'Sample Weight:' in content_peek:
            logger.info('Routing to PerkinElmer DSC schema.')
            entry = DSCMeasurement()
        else:
            logger.error(f'Unrecognized thermal file format: {filename}')
            return

        entry.data_file = filename
        archive.data = entry

        entry.normalize(archive, logger)