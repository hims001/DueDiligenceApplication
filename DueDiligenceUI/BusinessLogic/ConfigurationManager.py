from configparser import ConfigParser

config_path = 'DueDiligenceUI/BusinessLogic/app.config'
parser = ConfigParser()
parser.read(config_path)

def read_config(config_name):
    # for section_name in parser.sections():
    #     print('Section:', section_name)
    #     print('  Options:', parser.options(section_name))
    #     for key, value in parser.items(section_name):
    #         print('  {} = {}'.format(key, value))
    #     print()
    return parser.get('default_config', config_name)

def write_config(config_name, value):
    parser.set('default_config', config_name, value)
    with open(config_path, 'w') as configfile:
        parser.write(configfile)