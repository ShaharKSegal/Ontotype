def lazy_init(obj, attr: str, attr_getter):
    if not hasattr(obj, attr) or getattr(obj, attr) is None:
        setattr(obj, attr, attr_getter())
    return getattr(obj, attr)