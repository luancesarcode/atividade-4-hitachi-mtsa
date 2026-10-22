from importlib import import_module

__all__ = [
    'TRCVAE',
    'GANF',
    'RANSynCoders',
    'OSVM',
    'IForest',
    'Transformer',
    'Hitachi',
    'HitachiDCASE2020',
    'MFCCMix',
    'BARNNLSTM',
    'calculate_aucroc',
    'calculate_confusion_matrix',
    'files_train_test_split',
    'files_train_test_split_dcase2020_task2',
    'files_train_test_split_combined',
    'build_signal_processing_steps',
    'Wav2Array',
    'Demux2Array',
    'AbnormalSplit',
    'NORMAL',
    'ABNORMAL',
    'TRVAE',
    'TRCVAES',
]

_NAME_TO_MODULE = {
    'TRCVAE': '.models',
    'GANF': '.models.hybrid',
    'RANSynCoders': '.models',
    'OSVM': '.models',
    'TRVAE': '.models',
    'IForest': '.models',
    'TRCVAES': '.models',
    'Transformer': '.models',
    'Hitachi': '.models',
    'HitachiDCASE2020': '.models',
    'MFCCMix': '.models',
    'BARNNLSTM': '.models',
    'calculate_aucroc': '.utils.metrics',
    'calculate_confusion_matrix': '.utils.metrics',
    'files_train_test_split': '.utils.utils',
    'files_train_test_split_dcase2020_task2': '.utils.utils',
    'files_train_test_split_combined': '.utils.utils',
    'build_signal_processing_steps': '.utils.utils',
    'Wav2Array': '.utils.utils',
    'Demux2Array': '.utils.utils',
    'AbnormalSplit': '.utils.utils',
    'NORMAL': '.utils.utils',
    'ABNORMAL': '.utils.utils',
}


def __getattr__(name):
    try:
        module_name = _NAME_TO_MODULE[name]
    except KeyError as exc:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exc
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
