from nomad.config.models.plugins import ParserEntryPoint


class ThermalParserEntryPoint(ParserEntryPoint):
    def load(self):
        from nomad_measurements_thermal.parsers.parser import ThermalParser

        return ThermalParser(**self.dict())


parser_entry_point = ThermalParserEntryPoint(
    name='ThermalParser',
    description='Parser for Thermal Analysis files.',
    mainfile_mime_re='text/.*',
    mainfile_name_re='.*\\.(dat)$',
)