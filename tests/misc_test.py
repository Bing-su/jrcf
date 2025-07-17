def test_version():
    from jrcf import __version__  # noqa: PLC0415

    assert isinstance(__version__, str)
    assert __version__ != "unknown"


def test_java_gc():
    from jrcf import java_gc  # noqa: PLC0415

    java_gc()
