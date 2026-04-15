from dataclasses import dataclass

from running_modes.configurations.log_configuration import LogConfiguration


@dataclass
class ConfigurationEnvelope:
    run_type: str
    parameters: dict
    logging: LogConfiguration = None

