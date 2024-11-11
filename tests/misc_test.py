def test_version():
    from jrcf import __version__

    assert isinstance(__version__, str)
    assert __version__ != "unknown"
