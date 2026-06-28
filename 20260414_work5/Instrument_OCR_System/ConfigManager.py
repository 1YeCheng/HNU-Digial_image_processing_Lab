import json

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_config(self):
        return self.config

    def get_params(self, section, keys):
        res = {}
        cfg = self.config.get(section, {})
        for k in keys:
            res[k] = cfg.get(k)
        return res