from flax.core.frozen_dict import freeze

ft_config = {
    'batch_size': 1,
    'pt_wts_file': None,
    'cmv_modes_dir': None,
    'n_epochs': 10,
}

ft_config = freeze(ft_config)