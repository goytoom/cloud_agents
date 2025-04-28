#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():

    if sys.argv[1] != "migrate":

        port = 8000 + int(sys.argv[2])
        cli_args = [sys.argv[0], 'runserver', f'0.0.0.0:{port}']

    else:
        cli_args = [sys.argv[0], 'migrate']

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'frontend_server.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(cli_args)


if __name__ == '__main__':
    main()
