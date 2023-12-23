# coding = utf-8
import yaml


# load yaml file
def get_config(config):
    with open(config, 'r') as stream:
        # 修改，新增Loader
        return yaml.load(stream, Loader=yaml.SafeLoader)

