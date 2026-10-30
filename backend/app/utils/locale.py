import json
import os
import threading
from flask import request, has_request_context

_thread_local = threading.local()

_locales_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'locales')

# Load language registry
with open(os.path.join(_locales_dir, 'languages.json'), 'r', encoding='utf-8') as f:
    _languages = json.load(f)

# Load translation files
_translations = {}
for filename in os.listdir(_locales_dir):
    if filename.endswith('.json') and filename != 'languages.json':
        locale_name = filename[:-5]
        with open(os.path.join(_locales_dir, filename), 'r', encoding='utf-8') as f:
            _translations[locale_name] = json.load(f)


_supported_locales = set(_languages) | set(_translations)
DEFAULT_LOCALE = os.environ.get('MIROFISH_DEFAULT_LOCALE', 'en').strip().lower()
if DEFAULT_LOCALE not in _supported_locales:
    DEFAULT_LOCALE = 'zh'


def resolve_locale(raw: str) -> str:
    """Resolve an Accept-Language value (e.g. "en-US,en;q=0.9") to a supported locale."""
    for part in (raw or '').split(','):
        tag = part.split(';')[0].strip().lower()
        if not tag:
            continue
        if tag in _supported_locales:
            return tag
        primary = tag.split('-')[0]
        if primary in _supported_locales:
            return primary
    return DEFAULT_LOCALE


def set_locale(locale: str):
    """Set locale for current thread. Call at the start of background threads."""
    _thread_local.locale = locale


def get_locale() -> str:
    if has_request_context():
        return resolve_locale(request.headers.get('Accept-Language', ''))
    return getattr(_thread_local, 'locale', DEFAULT_LOCALE)


def t(key: str, **kwargs) -> str:
    locale = get_locale()
    messages = _translations.get(locale, _translations.get('en', _translations.get('zh', {})))

    value = messages
    for part in key.split('.'):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break

    if value is None:
        value = _translations.get('zh', {})
        for part in key.split('.'):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break

    if value is None:
        return key

    if kwargs:
        for k, v in kwargs.items():
            value = value.replace(f'{{{k}}}', str(v))

    return value


def get_language_instruction() -> str:
    locale = get_locale()
    lang_config = _languages.get(locale, _languages.get(DEFAULT_LOCALE, {}))
    return lang_config.get('llmInstruction', '请使用中文回答。')
