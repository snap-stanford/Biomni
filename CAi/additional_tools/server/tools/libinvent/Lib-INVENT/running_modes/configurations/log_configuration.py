from dataclasses import dataclass


@dataclass
class LogConfiguration:
    logging_path: str
    sender: str = None
    recipient: str = "local"
    job_id: str = None
    job_name: str = "default_name"
