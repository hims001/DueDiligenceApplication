from configparser import ConfigParser

config_path = 'DueDiligenceUI/BusinessLogic/app.config'
parser = ConfigParser()
parser.read(config_path)


def read_config(config_name: str) -> str:
    """Reads config value from app.config"""
    return parser.get('default_config', config_name)


def write_config(config_name: str, value):
    """Writes config value to the app.config"""
    parser.set('default_config', config_name, value)
    with open(config_path, 'w') as configfile:
        parser.write(configfile)
