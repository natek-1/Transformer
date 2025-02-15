from transformer.entity.config_entity import DataIngestionConfig
from transformer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from transformer.utils.main_utils import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        #self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            save_dir=config.save_dir,
            datasource=config.datasource,
            lang_src=config.lang_src,
            lang_tgt=config.lang_tgt
        )

        return data_ingestion_config