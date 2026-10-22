from importlib import import_module

__all__ = [
    'BARNNLSTM',
    'GANF',
    'OSVM',
    'TRCVAE',
    'TRVAE',
    'TRCVAES'
    'Hitachi',
    'HitachiDCASE2020',
    'IForest',
    'MFCCMix',
    'RANSynCoders',
    'Transformer'
]

_NAME_TO_MODULE = {
    'Hitachi': '.hitachi',
    'HitachiDCASE2020': '.hitachi',
    'MFCCMix': '.mfccmix',
    'GANF': '.ganf',
    'RANSynCoders': '.ransyncorders',
    'IForest': '.isolationforest',
    'OSVM': '.oneClassSVM',
    'Transformer': '.transformer',
    'TRCVAE': '.trcvae',
    'BARNNLSTM': '.barnn_lstm',
    'TRVAE': '.trvae',
    'TRCVAES': '.trcvaes',
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
