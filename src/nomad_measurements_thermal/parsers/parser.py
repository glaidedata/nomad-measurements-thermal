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
        """Gatekeeper for Dilatometry .dat files."""
        if not super().is_mainfile(filename, mime, buffer, decoded_buffer, compression):
            return False

        # String check based on the structure of your Cu TEC VerAcc file
        if decoded_buffer and '[Header]' in decoded_buffer and '[Data]' in decoded_buffer:
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