_debugger_enabled = False


def enable_debugger(debug_ip="localhost", debug_port=8223):
    global _debugger_enabled
    if not _debugger_enabled:
        import pydevd_pycharm

        print("Debugging enabled")
        pydevd_pycharm.settrace(
            debug_ip,
            port=debug_port,
            stdoutToServer=True,
            stderrToServer=True,
        )
        _debugger_enabled = True
