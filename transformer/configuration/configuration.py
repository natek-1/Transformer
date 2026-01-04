from transformer.entity.config_entity import DataIngestionConfig, DataTransformationConfig
from transformer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from transformer.utils.main_utils import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

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

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_file=config.tokenizer_file,
            seq_len=self.params.model_config.seq_len,
            batch_size=self.params.model_config.batch_size,
            lang_src=config.lang_src if hasattr(config, 'lang_src') else self.config.data_ingestion.lang_src,
            lang_tgt=config.lang_tgt if hasattr(config, 'lang_tgt') else self.config.data_ingestion.lang_tgt
        )

        return data_transformation_config