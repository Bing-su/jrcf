def test_version():
    from jrcf import __version__

    assert isinstance(__version__, str)
    assert __version__ != "unknown"


def test_java_gc():
    from jrcf import java_gc

    java_gc()
