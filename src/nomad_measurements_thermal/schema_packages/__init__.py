from nomad.config.models.plugins import SchemaPackageEntryPoint
from pydantic import Field


class ThermalSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_measurements_thermal.schema_packages.schema_package import m_package

        return m_package


schema_package_entry_point = ThermalSchemaPackageEntryPoint(
    name='ThermalSchemaPackage',
    description='Schema package for Thermal Measurements (Dilatometry, DSC, ACS).',
)
