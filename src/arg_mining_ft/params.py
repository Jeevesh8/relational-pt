from frozendict import frozendict

ft_config = {
    'batch_size': 1,
    'pt_wts_file': None,
    'cmv_modes_dir': None,
    'n_epochs': 10,
}

ft_config = frozendict(ft_config)