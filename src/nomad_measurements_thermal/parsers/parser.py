from nomad.datamodel.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser

from nomad_measurements_thermal.schema_packages.schema_package import (
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
        """Gatekeeper for thermal-analysis .dat files."""
        if not super().is_mainfile(filename, mime, buffer, decoded_buffer, compression):
            return False

        if not decoded_buffer:
            return False

        # Keep the generic structural checks, but add thermal-specific markers to
        # avoid grabbing other Quantum Design `.dat` formats.
        has_sections = '[Header]' in decoded_buffer and '[Data]' in decoded_buffer
        has_thermal_marker = any(
            marker in decoded_buffer
            for marker in (
                'BEGIN:PARAMS',
                'dilation_offset',
                'cell_constant',
                'Therm Resistance',
                'Dilation (ppm)',
            )
        )

        if has_sections and has_thermal_marker:
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

        # Instantiate the Thermal schema
        entry = ThermalMeasurement()

        # Point the schema to the matched file
        entry.data_file = mainfile.rsplit('/', maxsplit=1)[-1]

        # Attach the schema to the archive
        archive.data = entry

        # Trigger the normalize function we wrote in schema_package.py to extract the arrays
        entry.normalize(archive, logger)
