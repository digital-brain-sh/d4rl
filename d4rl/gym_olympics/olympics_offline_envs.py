from argparse import Namespace

from .envs import football, rd_running, running, table_hockey, wrestling
from d4rl import offline_env


class Offline_rd_running(rd_running, offline_env.OfflineEnv):

    def __init__(self, **kwargs):
        rd_running.__init__(self)
        offline_env.OfflineEnv.__init__(
            self,
            dataset_url=kwargs.get("dataset_url", None),
            ref_max_score=kwargs.get("ref_max_score", None),
            ref_min_score=kwargs.get("ref_min_score", None),
            deprecated=kwargs.get("deprecated", False),
            deprecation_message=kwargs.get("deprecation_message", None)
        )

def get_rd_running(**kwargs):
    return Offline_rd_running(**kwargs)

class Offline_running(running, offline_env.OfflineEnv):

    def __init__(self, **kwargs):
        running.__init__(self, map_id=kwargs.get('map_id', None))
        offline_env.OfflineEnv.__init__(
            self,
            dataset_url=kwargs.get("dataset_url", None),
            ref_max_score=kwargs.get("ref_max_score", None),
            ref_min_score=kwargs.get("ref_min_score", None),
            deprecated=kwargs.get("deprecated", False),
            deprecation_message=kwargs.get("deprecation_message", None)
        )

def get_running(**kwargs):
    return Offline_running(**kwargs)

class Offline_football(football, offline_env.OfflineEnv):

    def __init__(self, **kwargs):
        football.__init__(self)
        offline_env.OfflineEnv.__init__(
            self,
            dataset_url=kwargs.get("dataset_url", None),
            ref_max_score=kwargs.get("ref_max_score", None),
            ref_min_score=kwargs.get("ref_min_score", None),
            deprecated=kwargs.get("deprecated", False),
            deprecation_message=kwargs.get("deprecation_message", None)
        )

def get_football(**kwargs):
    return Offline_football(**kwargs)

class Offline_table_hockey(table_hockey, offline_env.OfflineEnv):

    def __init__(self, **kwargs):
        table_hockey.__init__(self)
        offline_env.OfflineEnv.__init__(
            self,
            dataset_url=kwargs.get("dataset_url", None),
            ref_max_score=kwargs.get("ref_max_score", None),
            ref_min_score=kwargs.get("ref_min_score", None),
            deprecated=kwargs.get("deprecated", False),
            deprecation_message=kwargs.get("deprecation_message", None)
        )

def get_table_hockey(**kwargs):
    return Offline_table_hockey(**kwargs)

class Offline_wrestling(wrestling, offline_env.OfflineEnv):

    def __init__(self, **kwargs):
        wrestling.__init__(self)
        offline_env.OfflineEnv.__init__(
            self,
            dataset_url=kwargs.get("dataset_url", None),
            ref_max_score=kwargs.get("ref_max_score", None),
            ref_min_score=kwargs.get("ref_min_score", None),
            deprecated=kwargs.get("deprecated", False),
            deprecation_message=kwargs.get("deprecation_message", None)
        )

def get_wrestling(**kwargs):
    return Offline_wrestling(**kwargs)

